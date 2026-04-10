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

from calibration.threshold_search import find_threshold_for_target_coverage
from datasets.ep_operator_dataset import build_case_index
from models.backbones.fno import FNO2d
from models.heads.global_rejector import GlobalRejectorMLP
from models.heads.local_rejector import LocalRejectorCNN
from utils.hybrid_inference import HybridStats, masked_rollout_rmse, rollout_fno_only, rollout_hybrid_global, rollout_hybrid_local

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
    p = argparse.ArgumentParser(description="Week 12: local vs global hybrid @ matched coverage (ID val calibration).")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--fno-checkpoint", type=Path, required=True)
    p.add_argument("--global-checkpoint", type=Path, required=True)
    p.add_argument("--local-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week12_main"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch-stride", type=int, default=4, choices=[4, 8])
    p.add_argument("--coverages", type=str, default="0.5,0.65,0.8,0.9")
    p.add_argument("--oracle-equiv-per-case", type=float, default=None)
    p.add_argument("--max-val-cases", type=int, default=32, help="Cap val cases for local tau search speed.")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_keys(family: str, split_name: str) -> str:
    if family == "id":
        return f"{split_name}_case_ids"
    k = f"{split_name}_shift_case_ids"
    return k


def load_case_ids(split_path: Path, family: str, split_name: str) -> list[str]:
    s = read_json(split_path)
    k = _split_keys(family, split_name)
    ids = list(s.get(k, []))
    if not ids and split_name == "test":
        ids = list(s.get("test_case_ids", []))
    return ids


def mean_deferred_fraction_stats(stats_list: list[HybridStats]) -> float:
    if not stats_list:
        return 0.0
    return float(np.mean([s.deferred_fraction() for s in stats_list]))


def search_local_tau(
    fno: torch.nn.Module,
    loc: torch.nn.Module,
    val_ids: list[str],
    case_index: dict[str, Any],
    device: str,
    target_mean_defer: float,
) -> float:
    """Find prob threshold achieving mean per-case defer fraction ~= target_mean_defer on val."""

    def eval_tau(tau: float) -> float:
        fr: list[float] = []
        for cid in val_ids:
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
            m = torch.from_numpy(data["mask"].astype(np.float32)).to(device)
            params_t = torch.tensor([[diffusion, excitability, restitution]], device=device)
            dt_t = torch.tensor([dt], device=device)
            _, st = rollout_hybrid_local(fno, loc, v, r, m, params_t, dt_t, tau, device)
            fr.append(st.deferred_fraction())
        return float(np.mean(fr)) if fr else 0.0

    # Higher prob threshold => fewer pixels above threshold => lower defer rate (monotone decreasing).
    lo, hi = 0.0, 1.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if eval_tau(mid) > target_mean_defer:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eval_global_at_tau(
    case_ids: list[str],
    case_index: dict[str, Any],
    feature_map: dict[str, Any],
    fno: torch.nn.Module,
    glob: torch.nn.Module,
    tau: float,
    device: str,
) -> tuple[float, float, HybridStats]:
    rmses: list[float] = []
    slist: list[HybridStats] = []
    for cid in case_ids:
        info = case_index[cid]
        data = np.load(info.tensor_path)
        v, r, mask = data["V"], data["R"], data["mask"]
        gt = np.stack([v, r], axis=1).astype(np.float32)
        row = feature_map[cid]
        vec = torch.from_numpy(row[FEATURE_COLS].to_numpy(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            sc = float(glob(vec).cpu().item())
        pr, st = rollout_hybrid_global(fno, v, r, mask, sc, float(tau), device)
        rmses.append(masked_rollout_rmse(pr, gt, mask))
        slist.append(st)
    agg = HybridStats()
    for s in slist:
        agg.n_fno_forwards += s.n_fno_forwards
        agg.n_full_case_defers += s.n_full_case_defers
        agg.n_pixel_repairs += s.n_pixel_repairs
        agg.n_tissue_pixels += s.n_tissue_pixels
        agg.n_spacetime_pairs += s.n_spacetime_pairs
    return float(np.mean(rmses)), mean_deferred_fraction_stats(slist), agg


def eval_local_at_tau(
    case_ids: list[str],
    case_index: dict[str, Any],
    fno: torch.nn.Module,
    loc: torch.nn.Module,
    tau_p: float,
    device: str,
) -> tuple[float, float, HybridStats]:
    rmses: list[float] = []
    slist: list[HybridStats] = []
    for cid in case_ids:
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
        m = torch.from_numpy(data["mask"].astype(np.float32)).to(device)
        gt = np.stack([data["V"], data["R"]], axis=1).astype(np.float32)
        params_t = torch.tensor([[diffusion, excitability, restitution]], device=device)
        dt_t = torch.tensor([dt], device=device)
        pr, st = rollout_hybrid_local(fno, loc, v, r, m, params_t, dt_t, float(tau_p), device)
        rmses.append(masked_rollout_rmse(pr, gt, m.cpu().numpy()))
        slist.append(st)
    agg = HybridStats()
    for s in slist:
        agg.n_fno_forwards += s.n_fno_forwards
        agg.n_full_case_defers += s.n_full_case_defers
        agg.n_pixel_repairs += s.n_pixel_repairs
        agg.n_tissue_pixels += s.n_tissue_pixels
        agg.n_spacetime_pairs += s.n_spacetime_pairs
    return float(np.mean(rmses)), mean_deferred_fraction_stats(slist), agg


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    device = args.device
    manifest = read_json(args.manifest)
    case_index = build_case_index(args.manifest, repo_root)
    df = pd.read_csv(args.feature_csv)
    feature_map = {row["case_id"]: row for _, row in df.iterrows()}

    coverages = [float(x) for x in args.coverages.split(",")]

    fno = FNO2d(in_channels=2, out_channels=2, width=24, depth=3, modes_h=12, modes_w=12).to(device)
    fno.load_state_dict(torch.load(args.fno_checkpoint, map_location=device)["model_state_dict"])
    fno.eval()

    glob = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1).to(device)
    glob.load_state_dict(torch.load(args.global_checkpoint, map_location=device)["model_state_dict"])
    glob.eval()

    loc = LocalRejectorCNN(in_channels=5, hidden=64, patch_stride=args.patch_stride).to(device)
    loc.load_state_dict(torch.load(args.local_checkpoint, map_location=device)["model_state_dict"])
    loc.eval()

    split_id_path = Path(manifest["split_files"]["id"])
    val_ids_all = [c for c in load_case_ids(split_id_path, "id", "val") if c in feature_map and c in case_index]
    if args.max_val_cases:
        val_ids_all = val_ids_all[: args.max_val_cases]
    if not val_ids_all:
        raise SystemExit("No ID val cases with features — build features first or lower --max-val-cases.")

    xs_val = torch.from_numpy(
        np.stack([feature_map[c][FEATURE_COLS].to_numpy(np.float32) for c in val_ids_all], axis=0)
    ).to(device)
    with torch.no_grad():
        val_scores = glob(xs_val).cpu().numpy()

    Ts = [int(np.load(case_index[c].tensor_path)["V"].shape[0]) for c in case_index]
    median_t = int(np.median(Ts)) if Ts else 10
    oracle_equiv = args.oracle_equiv_per_case if args.oracle_equiv_per_case is not None else float(max(median_t - 1, 1))
    median_tissue = int(
        np.median([(np.load(case_index[c].tensor_path)["mask"] > 0).sum() for c in case_index])
    )
    pixels_per_forward = float(max(median_tissue, 1.0))

    family_map = {
        "id": manifest["split_files"]["id"],
        "parameter_shift": manifest["split_files"]["parameter_shift"],
        "geometry_shift": manifest["split_files"]["geometry_shift"],
        "long_rollout": manifest["split_files"]["long_rollout"],
    }

    rows: list[dict[str, Any]] = []

    for cov in tqdm(coverages, desc="coverage"):
        tau_g = float(find_threshold_for_target_coverage(val_scores, cov))
        target_defer = float(1.0 - cov)
        tau_l = search_local_tau(fno, loc, val_ids_all, case_index, device, target_defer)

        for family, split_str in family_map.items():
            test_ids = [
                c
                for c in load_case_ids(Path(split_str), family, "test")
                if c in feature_map and c in case_index
            ]
            if not test_ids:
                continue

            g_rmse, g_def, g_st = eval_global_at_tau(test_ids, case_index, feature_map, fno, glob, tau_g, device)
            l_rmse, l_def, l_st = eval_local_at_tau(test_ids, case_index, fno, loc, tau_l, device)

            baseline_preds = []
            for cid in test_ids:
                data = np.load(case_index[cid].tensor_path)
                v, r, mask = data["V"], data["R"], data["mask"]
                gt = np.stack([v, r], axis=1).astype(np.float32)
                pr, _ = rollout_fno_only(fno, v, r, device)
                baseline_preds.append(masked_rollout_rmse(pr, gt, mask))
            base_rmse = float(np.mean(baseline_preds))

            ref_fno = sum(int(np.load(case_index[c].tensor_path)["V"].shape[0]) - 1 for c in test_ids)
            compute_norm_g = g_st.compute_units(oracle_equiv, pixels_per_forward) / max(ref_fno, 1.0)
            compute_norm_l = l_st.compute_units(oracle_equiv, pixels_per_forward) / max(ref_fno, 1.0)

            rows.append(
                {
                    "target_coverage_cal": cov,
                    "family": family,
                    "tau_global": tau_g,
                    "tau_local_prob": tau_l,
                    "test_hybrid_rmse_global": g_rmse,
                    "test_hybrid_rmse_local": l_rmse,
                    "test_mean_defer_frac_global": g_def,
                    "test_mean_defer_frac_local": l_def,
                    "test_compute_norm_global": compute_norm_g,
                    "test_compute_norm_local": compute_norm_l,
                    "test_baseline_fno_rmse": base_rmse,
                    "local_beats_global_rmse": int(l_rmse < g_rmse),
                }
            )

    fields = sorted({k for r in rows for k in r.keys()})
    with (args.output_dir / "week12_main_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (args.output_dir / "week12_main_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Main figure: first coverage slice with all families
    if rows:
        cov0 = coverages[len(coverages) // 2]
        sub = [r for r in rows if abs(r["target_coverage_cal"] - cov0) < 1e-6]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sub))
        wbar = 0.35
        ax.bar(x - wbar / 2, [r["test_hybrid_rmse_global"] for r in sub], width=wbar, label="hybrid global")
        ax.bar(x + wbar / 2, [r["test_hybrid_rmse_local"] for r in sub], width=wbar, label="hybrid local")
        ax.set_xticks(x)
        ax.set_xticklabels([r["family"] for r in sub], rotation=15, ha="right")
        ax.set_ylabel("Hybrid rollout RMSE (test)")
        ax.set_title(f"Week 12 — local vs global @ matched val coverage ≈ {cov0}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "week12_main_bar_by_family.png", dpi=180)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for fam in family_map:
            pts = sorted(
                [
                    (r["test_compute_norm_local"], r["test_hybrid_rmse_local"])
                    for r in rows
                    if r["family"] == fam
                ]
            )
            if pts:
                ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", label=f"{fam} local")
            pts_g = sorted(
                [
                    (r["test_compute_norm_global"], r["test_hybrid_rmse_global"])
                    for r in rows
                    if r["family"] == fam
                ]
            )
            if pts_g:
                ax2.plot([p[0] for p in pts_g], [p[1] for p in pts_g], marker="o", linestyle="--", label=f"{fam} glob")
        ax2.set_xlabel("Normalized compute vs same-split FNO-only")
        ax2.set_ylabel("Hybrid RMSE")
        ax2.set_title("Week 12 — risk vs compute (matched calibration)")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(args.output_dir / "week12_risk_compute_traces.png", dpi=180)
        plt.close(fig2)

    card = [
        "# Week 12 — Local vs global hybrid",
        "",
        "- Calibration: global τ from **ID val** coverage; local τ_prob via bisection to match **mean defer fraction** (1−coverage) on **ID val**.",
        "- Evaluation: same τ settings applied to each **test** split (including shifts).",
        f"- Oracle equiv (full-case defer): `{oracle_equiv}` FNO-step units",
        "",
        f"- Table: `week12_main_table.csv` ({len(rows)} rows).",
        f"- Count (local RMSE < global RMSE): {sum(r.get('local_beats_global_rmse', 0) for r in rows)}/{len(rows)} row-wise",
    ]
    (args.output_dir / "week12_main_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[OK] Week 12 main: {args.output_dir}")


if __name__ == "__main__":
    main()
