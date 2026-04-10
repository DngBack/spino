#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index
from losses.physics_loss import pde_residual_magnitude_map
from models.backbones.fno import FNO2d
from models.heads.global_rejector import GlobalRejectorMLP
from models.heads.local_rejector import LocalRejectorCNN


FEATURE_COLS = [
    "residual_mean",
    "residual_max",
    "uncertainty_mean",
    "uncertainty_max",
    "drift_mean",
    "drift_slope",
    "ood_centroid_distance",
    "ood_nn_distance",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week10 oracle studies: global/local headroom.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--feature-csv", type=Path, default=Path("outputs/week6_features/fno_20260409/global_reliability_features.csv"))
    p.add_argument("--global-checkpoint", type=Path, default=Path("outputs/week7_global_rejector/20260409_210702/best_global_rejector.pt"))
    p.add_argument("--fno-checkpoint", type=Path, default=Path("outputs/week3_fno/20260409_202231/best_fno.pt"))
    p.add_argument("--local-checkpoint", type=Path, default=Path("outputs/week9_local_rejector/20260409_211421/best_local_rejector.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week10_oracle"))
    p.add_argument("--patch-stride", type=int, default=4, choices=[4, 8])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-coverage", type=float, default=0.75)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def select_ids(split_payload: dict[str, Any], family: str) -> list[str]:
    if family == "id":
        return list(split_payload.get("test_case_ids", []))
    return list(split_payload.get("test_shift_case_ids", []) or split_payload.get("test_case_ids", []))


def selective_risk_from_scores(scores: np.ndarray, risks: np.ndarray, coverage: float) -> float:
    tau = np.quantile(scores, coverage)
    accepted = scores <= tau
    if not np.any(accepted):
        return float("inf")
    return float(np.mean(risks[accepted]))


def selective_risk_oracle_global(risks: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(len(risks) * coverage)))
    idx = np.argsort(risks)[:k]
    return float(np.mean(risks[idx]))


def local_patch_arrays_for_family(
    family_ids: list[str],
    case_index: dict[str, Any],
    fno: torch.nn.Module,
    local: torch.nn.Module,
    patch_stride: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns concatenated arrays:
      patch_err: [Npatch]
      local_score: [Npatch]
    """
    patch_err_all: list[np.ndarray] = []
    local_score_all: list[np.ndarray] = []

    for cid in tqdm(family_ids, desc="Local oracle prep", leave=False):
        info = case_index[cid]
        npz = np.load(info.tensor_path)
        meta = read_json(info.metadata_path)
        V = npz["V"].astype(np.float32)
        R = npz["R"].astype(np.float32)
        mask = npz["mask"].astype(np.float32)
        diffusion = float(meta["parameters"]["diffusion"]) * float(meta["parameters"]["conductivity"])
        excitability = float(meta["parameters"]["excitability"])
        restitution = float(meta["parameters"]["restitution"])
        dt = float(meta["time"]["dt"])

        vt = torch.from_numpy(V).to(device)
        rt = torch.from_numpy(R).to(device)
        m = torch.from_numpy(mask).to(device)
        params_t = torch.tensor([[diffusion, excitability, restitution]], dtype=torch.float32, device=device)
        dt_t = torch.tensor([dt], dtype=torch.float32, device=device)

        for t in range(V.shape[0] - 1):
            x = torch.stack([vt[t], rt[t]], dim=0).unsqueeze(0)
            y = torch.stack([vt[t + 1], rt[t + 1]], dim=0).unsqueeze(0)
            with torch.no_grad():
                pred = fno(x)
                res_map = pde_residual_magnitude_map(x, pred, params_t, dt_t, m.unsqueeze(0))
                inp = torch.cat([x, pred, res_map], dim=1)
                logits = local(inp)  # [1,1,hp,wp]
            patch_err = F.avg_pool2d((pred - y).abs().mean(dim=1, keepdim=True), kernel_size=patch_stride, stride=patch_stride)
            mask_p = F.avg_pool2d(m.unsqueeze(0).unsqueeze(0), kernel_size=patch_stride, stride=patch_stride)
            valid = (mask_p > 1e-3).cpu().numpy()[0, 0]
            pe = patch_err.cpu().numpy()[0, 0][valid]
            sc = logits.cpu().numpy()[0, 0][valid]
            if pe.size > 0:
                patch_err_all.append(pe)
                local_score_all.append(sc)

    if not patch_err_all:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(patch_err_all).astype(np.float32), np.concatenate(local_score_all).astype(np.float32)


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.manifest)
    feature_df = pd.read_csv(args.feature_csv)
    case_index = build_case_index(args.manifest)
    family_to_split = {
        "id": manifest["split_files"]["id"],
        "parameter_shift": manifest["split_files"]["parameter_shift"],
        "geometry_shift": manifest["split_files"]["geometry_shift"],
        "long_rollout": manifest["split_files"]["long_rollout"],
    }

    # Models
    fno = FNO2d(in_channels=2, out_channels=2, width=24, depth=3, modes_h=12, modes_w=12).to(args.device)
    fno.load_state_dict(torch.load(args.fno_checkpoint, map_location=args.device)["model_state_dict"])
    fno.eval()

    local = LocalRejectorCNN(in_channels=5, hidden=64, patch_stride=args.patch_stride).to(args.device)
    local.load_state_dict(torch.load(args.local_checkpoint, map_location=args.device)["model_state_dict"])
    local.eval()

    global_model = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1).to(args.device)
    global_model.load_state_dict(torch.load(args.global_checkpoint, map_location=args.device)["model_state_dict"])
    global_model.eval()

    coverages = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.9], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    bottleneck_rows: list[dict[str, Any]] = []

    for family, split_path in family_to_split.items():
        ids = select_ids(read_json(Path(split_path)), family)
        sdf = feature_df[feature_df["case_id"].isin(ids)].copy()
        if sdf.empty:
            continue

        # Global learned/oracle
        with torch.no_grad():
            gs = global_model(torch.from_numpy(sdf[FEATURE_COLS].to_numpy(np.float32)).to(args.device)).cpu().numpy()
        gr = sdf["rollout_rmse"].to_numpy(np.float32)

        global_learn_curve, global_oracle_curve = [], []
        for c in coverages:
            gl = selective_risk_from_scores(gs, gr, float(c))
            go = selective_risk_oracle_global(gr, float(c))
            global_learn_curve.append(gl)
            global_oracle_curve.append(go)
            rows.append({"family": family, "selector_type": "global_learned", "coverage": float(c), "selective_risk": gl})
            rows.append({"family": family, "selector_type": "global_oracle", "coverage": float(c), "selective_risk": go})

        # Local learned/oracle + oracle repair upper bound
        patch_err, local_scores = local_patch_arrays_for_family(ids, case_index, fno, local, args.patch_stride, args.device)
        if patch_err.size == 0:
            continue
        local_learn_curve, local_oracle_curve = [], []
        local_hybrid_learn, local_hybrid_oracle = [], []
        for c in coverages:
            tau_l = np.quantile(local_scores, float(c))
            acc_l = local_scores <= tau_l
            tau_o = np.quantile(patch_err, float(c))  # oracle accepts lowest-error patches
            acc_o = patch_err <= tau_o
            lr = float(np.mean(patch_err[acc_l])) if np.any(acc_l) else float("inf")
            lo = float(np.mean(patch_err[acc_o])) if np.any(acc_o) else float("inf")
            # oracle repair: rejected patches replaced by GT => zero error on rejected
            # hybrid overall error over all patches = mean(err * accepted_mask)
            lh = float(np.mean(patch_err * acc_l.astype(np.float32)))
            oh = float(np.mean(patch_err * acc_o.astype(np.float32)))
            local_learn_curve.append(lr)
            local_oracle_curve.append(lo)
            local_hybrid_learn.append(lh)
            local_hybrid_oracle.append(oh)
            rows.append({"family": family, "selector_type": "local_learned", "coverage": float(c), "selective_risk": lr})
            rows.append({"family": family, "selector_type": "local_oracle", "coverage": float(c), "selective_risk": lo})
            rows.append({"family": family, "selector_type": "local_hybrid_learned", "coverage": float(c), "selective_risk": lh})
            rows.append({"family": family, "selector_type": "local_hybrid_oracle", "coverage": float(c), "selective_risk": oh})

        # Bottleneck diagnosis at target coverage
        c = float(args.target_coverage)
        g_l = selective_risk_from_scores(gs, gr, c)
        g_o = selective_risk_oracle_global(gr, c)
        tau_l = np.quantile(local_scores, c)
        acc_l = local_scores <= tau_l
        tau_o = np.quantile(patch_err, c)
        acc_o = patch_err <= tau_o
        l_l = float(np.mean(patch_err[acc_l]))
        l_o = float(np.mean(patch_err[acc_o]))
        bottleneck_rows.append(
            {
                "family": family,
                "target_coverage": c,
                "global_headroom": g_l - g_o,
                "local_headroom": l_l - l_o,
                "local_stronger_upside": (l_l - l_o) > (g_l - g_o),
            }
        )

        # Plots
        fig = plt.figure(figsize=(7, 4))
        plt.plot(coverages, global_learn_curve, marker="o", label="global learned")
        plt.plot(coverages, global_oracle_curve, marker="o", label="global oracle")
        plt.plot(coverages, local_learn_curve, marker="o", label="local learned")
        plt.plot(coverages, local_oracle_curve, marker="o", label="local oracle")
        plt.xlabel("Coverage")
        plt.ylabel("Selective risk")
        plt.title(f"Week10 Headroom ({family})")
        plt.grid(alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(run_dir / f"headroom_{family}.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 4))
        plt.plot(coverages, local_hybrid_learn, marker="o", label="local hybrid learned")
        plt.plot(coverages, local_hybrid_oracle, marker="o", label="local hybrid oracle")
        plt.xlabel("Coverage")
        plt.ylabel("Hybrid overall error (oracle repair)")
        plt.title(f"Oracle Repair Gain ({family})")
        plt.grid(alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(run_dir / f"oracle_repair_gain_{family}.png", dpi=180)
        plt.close(fig)

    write_csv(run_dir / "week10_oracle_comparison.csv", rows)
    write_csv(run_dir / "week10_bottleneck_diagnosis.csv", bottleneck_rows)
    (run_dir / "week10_oracle_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (run_dir / "week10_bottleneck_diagnosis.json").write_text(json.dumps(bottleneck_rows, indent=2), encoding="utf-8")

    card = [
        "# Week 10 Oracle Studies",
        "",
        f"- Target coverage for diagnosis: `{args.target_coverage}`",
        "- `global_headroom = learned selective risk - oracle selective risk`",
        "- `local_headroom = learned local selective risk - oracle local selective risk`",
        "",
        "## Bottleneck diagnosis",
    ]
    for r in bottleneck_rows:
        card.append(
            f"- `{r['family']}`: global_headroom={r['global_headroom']:.6f}, "
            f"local_headroom={r['local_headroom']:.6f}, local_stronger_upside={r['local_stronger_upside']}"
        )
    (run_dir / "week10_oracle_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[OK] Week10 oracle outputs: {run_dir}")


if __name__ == "__main__":
    main()
