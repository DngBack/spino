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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.threshold_search import (
    find_threshold_for_target_coverage,
    risk_coverage_curve,
)
from models.heads.global_rejector import GlobalRejectorMLP
from utils.numpy_compat import trapz_xy


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
    p = argparse.ArgumentParser(description="Evaluate global rejector on ID + shift families.")
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--rejector-checkpoint", type=Path, required=True)
    p.add_argument("--target-coverage", type=float, default=0.8)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week7_global_eval"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_ids(split_path: Path, family: str) -> list[str]:
    s = read_json(split_path)
    if family == "id":
        return list(s.get("test_case_ids", []))
    return list(s.get("test_shift_case_ids", []) or s.get("test_case_ids", []))


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.feature_csv)
    manifest = read_json(args.manifest)

    model = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1).to(args.device)
    ckpt = torch.load(args.rejector_checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Calibrate tau on ID validation set.
    split_id = Path(manifest["split_files"]["id"])
    val_ids = set(read_json(split_id).get("val_case_ids", []))
    val_df = df[df["case_id"].isin(val_ids)].copy()
    x_val = torch.from_numpy(val_df[FEATURE_COLS].to_numpy(np.float32, copy=True)).to(args.device)
    with torch.no_grad():
        val_scores = model(x_val).cpu().numpy()
    tau = find_threshold_for_target_coverage(val_scores, args.target_coverage)

    summary_rows = []
    per_case_dir = args.output_dir / "per_split"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    family_map = {
        "id": manifest["split_files"]["id"],
        "parameter_shift": manifest["split_files"]["parameter_shift"],
        "geometry_shift": manifest["split_files"]["geometry_shift"],
        "long_rollout": manifest["split_files"]["long_rollout"],
    }

    for fam, split_path_str in family_map.items():
        ids = get_ids(Path(split_path_str), fam)
        sdf = df[df["case_id"].isin(ids)].copy()
        if sdf.empty:
            continue
        x = torch.from_numpy(sdf[FEATURE_COLS].to_numpy(np.float32, copy=True)).to(args.device)
        with torch.no_grad():
            scores = model(x).cpu().numpy()
        risks = sdf["rollout_rmse"].to_numpy(np.float32)
        accepted = scores <= tau
        coverage = float(np.mean(accepted))
        selective_risk = float(np.mean(risks[accepted])) if np.any(accepted) else float("inf")
        overall_risk = float(np.mean(risks))
        _, covs, rsks = risk_coverage_curve(scores, risks, num_points=40)
        m = np.isfinite(rsks)
        aurc = trapz_xy(rsks[m], covs[m]) if np.any(m) else float("inf")

        row = {
            "family": fam,
            "num_cases": int(len(sdf)),
            "tau": float(tau),
            "coverage": coverage,
            "selective_risk": selective_risk,
            "overall_risk": overall_risk,
            "aurc": aurc,
        }
        summary_rows.append(row)
        sdf_out = sdf[["case_id", "rollout_rmse"]].copy()
        sdf_out["rejector_score"] = scores
        sdf_out["accepted"] = accepted.astype(int)
        sdf_out.to_csv(per_case_dir / f"{fam}_per_case_selective.csv", index=False)

        fig = plt.figure(figsize=(6, 4))
        plt.plot(covs, rsks, label=fam)
        plt.scatter([coverage], [selective_risk], c="red", s=40, label=f"tau@cov={args.target_coverage}")
        plt.xlabel("Coverage")
        plt.ylabel("Selective risk (rollout RMSE)")
        plt.title(f"Risk-Coverage: {fam}")
        plt.grid(alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"risk_coverage_{fam}.png", dpi=180)
        plt.close(fig)

    write_csv(args.output_dir / "global_selective_summary.csv", summary_rows)
    (args.output_dir / "global_selective_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    card = [
        "# Week 7 Global Selective Evaluation",
        "",
        f"- Target coverage (calibration): `{args.target_coverage}`",
        f"- Calibrated tau (ID val): `{tau:.6f}`",
        "",
        "## Split results",
    ]
    for r in summary_rows:
        card.append(
            f"- `{r['family']}`: coverage={r['coverage']:.3f}, "
            f"selective_risk={r['selective_risk']:.5f}, overall_risk={r['overall_risk']:.5f}, aurc={r['aurc']:.5f}"
        )
    (args.output_dir / "week7_global_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[OK] Week7 global eval outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
