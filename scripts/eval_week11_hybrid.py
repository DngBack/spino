#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index
from models.backbones.fno import FNO2d
from models.heads.global_rejector import GlobalRejectorMLP
from models.heads.local_rejector import LocalRejectorCNN
from utils.hybrid_inference import (
    HybridStats,
    masked_rollout_rmse,
    rollout_fno_only,
    rollout_hybrid_global,
    rollout_hybrid_local,
)

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
    p = argparse.ArgumentParser(description="Week 11: hybrid inference Pareto (compute vs risk).")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--fno-checkpoint", type=Path, required=True)
    p.add_argument("--global-checkpoint", type=Path, required=True)
    p.add_argument("--local-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week11_hybrid"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch-stride", type=int, default=4, choices=[4, 8])
    p.add_argument(
        "--oracle-equiv-per-case",
        type=float,
        default=None,
        help="Fallback cost in FNO-step units for a fully deferred case (default: median T-1).",
    )
    p.add_argument("--time-sync-cuda", action="store_true")
    p.add_argument("--max-cases-per-family", type=int, default=None)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_test_ids(split_path: Path, family: str) -> list[str]:
    s = read_json(split_path)
    if family == "id":
        return list(s.get("test_case_ids", []))
    return list(s.get("test_shift_case_ids", []) or s.get("test_case_ids", []))


def aggregate_stats(stats_list: list[HybridStats]) -> HybridStats:
    out = HybridStats()
    for s in stats_list:
        out.n_fno_forwards += s.n_fno_forwards
        out.n_full_case_defers += s.n_full_case_defers
        out.n_pixel_repairs += s.n_pixel_repairs
        out.n_tissue_pixels += s.n_tissue_pixels
        out.n_spacetime_pairs += s.n_spacetime_pairs
        out.wall_s_forward += s.wall_s_forward
        out.wall_s_merge += s.wall_s_merge
    return out


def mean_deferred_fraction(stats_list: list[HybridStats]) -> float:
    if not stats_list:
        return 0.0
    return float(np.mean([s.deferred_fraction() for s in stats_list]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    manifest = read_json(args.manifest)
    case_index = build_case_index(args.manifest, Path(__file__).resolve().parents[1])
    df = pd.read_csv(args.feature_csv)
    feature_map = {row["case_id"]: row for _, row in df.iterrows()}

    fno = FNO2d(in_channels=2, out_channels=2, width=24, depth=3, modes_h=12, modes_w=12).to(device)
    fno.load_state_dict(torch.load(args.fno_checkpoint, map_location=device)["model_state_dict"])
    fno.eval()

    glob = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1).to(device)
    glob.load_state_dict(torch.load(args.global_checkpoint, map_location=device)["model_state_dict"])
    glob.eval()

    loc = LocalRejectorCNN(in_channels=5, hidden=64, patch_stride=args.patch_stride).to(device)
    loc.load_state_dict(torch.load(args.local_checkpoint, map_location=device)["model_state_dict"])
    loc.eval()

    family_map = {
        "id": manifest["split_files"]["id"],
        "parameter_shift": manifest["split_files"]["parameter_shift"],
        "geometry_shift": manifest["split_files"]["geometry_shift"],
        "long_rollout": manifest["split_files"]["long_rollout"],
    }

    # Median horizon for oracle-equivalent default
    Ts = []
    for cid, info in case_index.items():
        data = np.load(info.tensor_path)
        Ts.append(int(data["V"].shape[0]))
    median_t = int(np.median(Ts)) if Ts else 10
    oracle_equiv = args.oracle_equiv_per_case if args.oracle_equiv_per_case is not None else float(max(median_t - 1, 1))
    median_tissue = int(np.median([(np.load(case_index[c].tensor_path)["mask"] > 0).sum() for c in case_index]))

    all_rows: list[dict[str, Any]] = []

    for family, split_str in family_map.items():
        case_ids = get_test_ids(Path(split_str), family)
        case_ids = [c for c in case_ids if c in feature_map and c in case_index]
        if args.max_cases_per_family is not None:
            case_ids = case_ids[: args.max_cases_per_family]
        if not case_ids:
            continue

        baseline_rmses: list[float] = []
        for cid in case_ids:
            info = case_index[cid]
            data = np.load(info.tensor_path)
            v, r, mask = data["V"], data["R"], data["mask"]
            gt = np.stack([v, r], axis=1).astype(np.float32)
            pr, _ = rollout_fno_only(fno, v, r, device, sync_cuda=args.time_sync_cuda)
            baseline_rmses.append(masked_rollout_rmse(pr, gt, mask))

        baseline_rmse_mean = float(np.mean(baseline_rmses))
        baseline_fno = sum(int(np.load(case_index[c].tensor_path)["V"].shape[0]) - 1 for c in case_ids)

        # Always defer: GT trajectory — RMSE 0, compute oracle
        st_def = HybridStats(
            n_fno_forwards=0,
            n_full_case_defers=len(case_ids),
            n_pixel_repairs=0,
            n_tissue_pixels=sum(int((np.load(case_index[c].tensor_path)["mask"] > 0).sum()) for c in case_ids),
            n_spacetime_pairs=sum(int(np.load(case_index[c].tensor_path)["V"].shape[0]) - 1 for c in case_ids),
        )
        cu_def = st_def.compute_units(oracle_equiv, float(median_tissue))
        all_rows.append(
            {
                "family": family,
                "policy": "always_defer",
                "param": None,
                "hybrid_rmse_mean": 0.0,
                "baseline_rmse_mean": baseline_rmse_mean,
                "deferred_frac": 1.0,
                "compute_units": cu_def,
                "n_fno": 0,
                "n_full_defers": st_def.n_full_case_defers,
                "wall_s_forward": 0.0,
            }
        )

        # Always predict
        st_pred = HybridStats(n_fno_forwards=baseline_fno)
        for cid in case_ids:
            T = int(np.load(case_index[cid].tensor_path)["V"].shape[0])
            st_pred.n_spacetime_pairs += T - 1
            st_pred.n_tissue_pixels += int((np.load(case_index[cid].tensor_path)["mask"] > 0).sum())
        cu_pred = st_pred.compute_units(oracle_equiv, float(median_tissue))
        all_rows.append(
            {
                "family": family,
                "policy": "always_predict",
                "param": None,
                "hybrid_rmse_mean": baseline_rmse_mean,
                "baseline_rmse_mean": baseline_rmse_mean,
                "deferred_frac": 0.0,
                "compute_units": cu_pred,
                "n_fno": baseline_fno,
                "n_full_defers": 0,
                "wall_s_forward": st_pred.wall_s_forward,
            }
        )

        # Global hybrid: sweep tau quantiles on test scores (analysis curve; calibrate on val in week 12)
        xs = []
        for cid in case_ids:
            row = feature_map[cid]
            vec = torch.from_numpy(row[FEATURE_COLS].to_numpy(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                xs.append(float(glob(vec).cpu().item()))
        xs_arr = np.array(xs, dtype=np.float32)
        taus_g = np.quantile(xs_arr, np.linspace(0.05, 0.95, 12))

        for tau in taus_g:
            rmses: list[float] = []
            slist: list[HybridStats] = []
            t_wall = 0.0
            for cid in tqdm(case_ids, desc=f"{family} global tau={tau:.3f}", leave=False):
                info = case_index[cid]
                data = np.load(info.tensor_path)
                v, r, mask = data["V"], data["R"], data["mask"]
                gt = np.stack([v, r], axis=1).astype(np.float32)
                row = feature_map[cid]
                vec = torch.from_numpy(row[FEATURE_COLS].to_numpy(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    sc = float(glob(vec).cpu().item())
                t0 = time.perf_counter()
                pr, st = rollout_hybrid_global(fno, v, r, mask, sc, float(tau), device, sync_cuda=args.time_sync_cuda)
                t_wall += time.perf_counter() - t0
                rmses.append(masked_rollout_rmse(pr, gt, mask))
                slist.append(st)
            agg = aggregate_stats(slist)
            all_rows.append(
                {
                    "family": family,
                    "policy": "hybrid_global",
                    "param": float(tau),
                    "hybrid_rmse_mean": float(np.mean(rmses)),
                    "baseline_rmse_mean": baseline_rmse_mean,
                    "deferred_frac": mean_deferred_fraction(slist),
                    "compute_units": agg.compute_units(oracle_equiv, float(median_tissue)),
                    "n_fno": agg.n_fno_forwards,
                    "n_full_defers": agg.n_full_case_defers,
                    "wall_s_forward": t_wall,
                }
            )

        # Local hybrid: sweep probability thresholds
        for tau_p in np.linspace(0.05, 0.95, 10):
            rmses = []
            slist = []
            t_wall = 0.0
            for cid in tqdm(case_ids, desc=f"{family} local p={tau_p:.2f}", leave=False):
                info = case_index[cid]
                data = np.load(info.tensor_path)
                meta = json.loads(info.metadata_path.read_text(encoding="utf-8"))
                params = meta["parameters"]
                diffusion = float(params["diffusion"]) * float(params["conductivity"])
                excitability = float(params["excitability"])
                restitution = float(params["restitution"])
                dt = float(meta["time"]["dt"])
                v = torch.from_numpy(data["V"].astype(np.float32)).to(device)
                r = torch.from_numpy(data["R"].astype(np.float32)).to(device)
                mask = torch.from_numpy(data["mask"].astype(np.float32)).to(device)
                gt = np.stack([data["V"], data["R"]], axis=1).astype(np.float32)
                params_t = torch.tensor([[diffusion, excitability, restitution]], device=device)
                dt_t = torch.tensor([dt], device=device)
                t0 = time.perf_counter()
                pr, st = rollout_hybrid_local(
                    fno, loc, v, r, mask, params_t, dt_t, float(tau_p), device, sync_cuda=args.time_sync_cuda
                )
                t_wall += time.perf_counter() - t0
                rmses.append(masked_rollout_rmse(pr, gt, mask.cpu().numpy()))
                slist.append(st)
            agg = aggregate_stats(slist)
            all_rows.append(
                {
                    "family": family,
                    "policy": "hybrid_local",
                    "param": float(tau_p),
                    "hybrid_rmse_mean": float(np.mean(rmses)),
                    "baseline_rmse_mean": baseline_rmse_mean,
                    "deferred_frac": mean_deferred_fraction(slist),
                    "compute_units": agg.compute_units(oracle_equiv, float(median_tissue)),
                    "n_fno": agg.n_fno_forwards,
                    "n_full_defers": agg.n_full_case_defers,
                    "wall_s_forward": t_wall,
                }
            )

        # Figure: hybrid vs compute for this family
        sub = [r for r in all_rows if r["family"] == family and r["policy"] in {"hybrid_global", "hybrid_local"}]
        if sub:
            fig, ax = plt.subplots(figsize=(6, 4))
            for pol, m in [("hybrid_global", "o"), ("hybrid_local", "s")]:
                pts = sorted([(r["compute_units"], r["hybrid_rmse_mean"]) for r in sub if r["policy"] == pol])
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, label=pol)
            ref_row = next(r for r in all_rows if r["family"] == family and r["policy"] == "always_predict")
            ax.scatter(
                [ref_row["compute_units"]],
                [baseline_rmse_mean],
                c="gray",
                marker="x",
                s=60,
                label="always_predict (ref)",
                zorder=5,
            )
            ax.set_xlabel("Compute units (FNO steps + scaled fallbacks)")
            ax.set_ylabel("Mean hybrid rollout RMSE")
            ax.set_title(f"Week 11 hybrid Pareto — {family}")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output_dir / f"pareto_compute_risk_{family}.png", dpi=180)
            plt.close(fig)

    # Normalize compute per family to always_predict = 1
    fields = sorted({k for r in all_rows for k in r.keys()})
    for fam in family_map:
        ref = next((r["compute_units"] for r in all_rows if r["family"] == fam and r["policy"] == "always_predict"), 1.0)
        for r in all_rows:
            if r["family"] == fam:
                r["compute_norm"] = float(r["compute_units"] / max(ref, 1e-8))

    with (args.output_dir / "week11_hybrid_pareto.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r.keys()}))
        w.writeheader()
        w.writerows(all_rows)
    (args.output_dir / "week11_hybrid_pareto.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    card = [
        "# Week 11 — Hybrid routing",
        "",
        f"- Oracle equiv / full defer: `{oracle_equiv}` FNO-step units",
        f"- Median tissue pixels (for scaling): `{median_tissue}`",
        "",
        "Outputs: `week11_hybrid_pareto.csv`, per-family `pareto_compute_risk_*.png`.",
    ]
    (args.output_dir / "week11_hybrid_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[OK] Week 11 hybrid: {args.output_dir}")


if __name__ == "__main__":
    main()
