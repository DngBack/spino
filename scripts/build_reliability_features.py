#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index, load_case_rollout, load_split_case_ids
from evaluators.predictive_metrics import rmse
from features.ood_features import extract_case_embedding, ood_distance_scores
from features.residual_maps import pde_residual_map
from features.rollout_drift import compute_rollout_drift_map
from features.uncertainty_features import perturbation_ensemble_variance
from models.backbones.fno import FNO2d
from models.backbones.pino_fno import PINOFNO2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reliability feature cache and analysis report.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--id-split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-type", choices=["fno", "pino"], default="fno")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week6_features"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/processed/features_cache"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--uncertainty-samples", type=int, default=6)
    p.add_argument("--uncertainty-noise-std", type=float, default=0.01)
    p.add_argument("--drift-horizon", type=int, default=40)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_one_step(model: torch.nn.Module, x_np: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(x_np[None, ...]).to(device)
        y = model(x).cpu().numpy()[0]
    return y.astype(np.float32)


def rollout_error(model: torch.nn.Module, gt: np.ndarray, device: str) -> float:
    cur = gt[0:1].copy()
    preds = [cur[0]]
    with torch.no_grad():
        for _ in range(gt.shape[0] - 1):
            y = model(torch.from_numpy(cur).to(device)).cpu().numpy()
            cur = y
            preds.append(cur[0])
    pred = np.stack(preds, axis=0)
    return rmse(pred, gt)


def save_case_cache(path: Path, residual_map: np.ndarray, uncertainty_map: np.ndarray, drift_map: np.ndarray, ood_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        residual_map=residual_map.astype(np.float32),
        uncertainty_map=uncertainty_map.astype(np.float32),
        drift_map=drift_map.astype(np.float32),
        ood_map=ood_map.astype(np.float32),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cls = FNO2d if args.model_type == "fno" else PINOFNO2d
    model = model_cls(
        in_channels=2,
        out_channels=2,
        width=args.width,
        depth=args.depth,
        modes_h=args.modes,
        modes_w=args.modes,
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    case_index = build_case_index(args.manifest)
    manifest = read_json(args.manifest)
    all_case_ids = [c["case_id"] for c in manifest["cases"]]
    train_case_ids = load_split_case_ids(args.id_split, "train")

    # Build train embedding bank for OOD features.
    train_embs = []
    for cid in train_case_ids:
        case = load_case_rollout(case_index[cid])
        x0 = np.stack([case["V"][0], case["R"][0]], axis=0)
        train_embs.append(extract_case_embedding(model, x0, args.device))
    train_embs_np = np.stack(train_embs, axis=0).astype(np.float32)

    run_cache = args.cache_dir / f"{args.model_type}_{Path(args.checkpoint).parent.name}"
    run_cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for cid in tqdm(all_case_ids, desc="Extract reliability features"):
        case = load_case_rollout(case_index[cid])
        v = case["V"]
        r = case["R"]
        mask = case["mask"]
        meta = case["metadata"]
        gt = np.stack([v, r], axis=1)  # [T,2,H,W]
        x0 = gt[0]
        emb = extract_case_embedding(model, x0, args.device)
        ood_scores = ood_distance_scores(train_embs_np, emb)

        # One-step residual map averaged over trajectory
        params = meta["parameters"]
        diffusion = float(params["diffusion"]) * float(params["conductivity"])
        excitability = float(params["excitability"])
        restitution = float(params["restitution"])
        dt = float(meta["time"]["dt"])

        residual_maps = []
        uncertainty_maps = []
        for t in range(gt.shape[0] - 1):
            x_t = gt[t]
            pred_t1 = predict_one_step(model, x_t, args.device)
            residual_maps.append(
                pde_residual_map(
                    v_t=x_t[0],
                    r_t=x_t[1],
                    v_t1_pred=pred_t1[0],
                    r_t1_pred=pred_t1[1],
                    diffusion=diffusion,
                    excitability=excitability,
                    restitution=restitution,
                    dt=dt,
                    mask=mask,
                )
            )
            u_map = perturbation_ensemble_variance(
                model,
                x=torch.from_numpy(x_t[None, ...]).to(args.device),
                num_samples=args.uncertainty_samples,
                noise_std=args.uncertainty_noise_std,
            )
            uncertainty_maps.append(u_map)

        residual_map = np.mean(np.stack(residual_maps, axis=0), axis=0).astype(np.float32)
        uncertainty_map = np.mean(np.stack(uncertainty_maps, axis=0), axis=0).astype(np.float32)
        drift_map, drift_slope = compute_rollout_drift_map(
            model=model, gt=gt.astype(np.float32), horizon=args.drift_horizon, device=args.device
        )
        ood_map = np.full_like(residual_map, fill_value=ood_scores["ood_centroid_distance"], dtype=np.float32)

        cache_path = run_cache / f"{cid}.npz"
        save_case_cache(cache_path, residual_map, uncertainty_map, drift_map, ood_map)

        roll_err = rollout_error(model, gt.astype(np.float32), args.device)
        rows.append(
            {
                "case_id": cid,
                "scenario_type": meta.get("scenario_type"),
                "geometry_id": meta.get("geometry_id"),
                "residual_mean": float(np.mean(residual_map)),
                "residual_max": float(np.max(residual_map)),
                "uncertainty_mean": float(np.mean(uncertainty_map)),
                "uncertainty_max": float(np.max(uncertainty_map)),
                "drift_mean": float(np.mean(drift_map)),
                "drift_slope": float(drift_slope),
                "ood_centroid_distance": float(ood_scores["ood_centroid_distance"]),
                "ood_nn_distance": float(ood_scores["ood_nn_distance"]),
                "rollout_rmse": float(roll_err),
                "feature_cache_path": str(cache_path),
            }
        )

    write_csv(out_dir / "global_reliability_features.csv", rows)
    (out_dir / "global_reliability_features.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    numeric_cols = [
        "residual_mean",
        "residual_max",
        "uncertainty_mean",
        "uncertainty_max",
        "drift_mean",
        "drift_slope",
        "ood_centroid_distance",
        "ood_nn_distance",
        "rollout_rmse",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    corr.to_csv(out_dir / "feature_correlation_matrix.csv")

    fig = plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Reliability Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation_matrix.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# Week 6 Feature Sanity Report",
        "",
        f"- Model type: `{args.model_type}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Cases processed: `{len(rows)}`",
        f"- Feature cache directory: `{run_cache}`",
        "",
        "## Mean feature values",
    ]
    for col in numeric_cols:
        report_lines.append(f"- `{col}`: `{df[col].mean():.6f}`")
    report_lines.extend(
        [
            "",
            "## Correlation with rollout_rmse",
        ]
    )
    for col in numeric_cols:
        if col == "rollout_rmse":
            continue
        report_lines.append(f"- `{col}` vs `rollout_rmse`: `{corr.loc[col, 'rollout_rmse']:.4f}`")
    (out_dir / "feature_sanity_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Feature cache: {run_cache}")
    print(f"[OK] Global feature table: {out_dir / 'global_reliability_features.csv'}")
    print(f"[OK] Correlation matrix: {out_dir / 'feature_correlation_matrix.csv'}")
    print(f"[OK] Sanity report: {out_dir / 'feature_sanity_report.md'}")


if __name__ == "__main__":
    main()
