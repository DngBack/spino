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

from calibration.threshold_search import find_threshold_for_target_coverage, risk_coverage_curve
from models.heads.global_rejector import GlobalRejectorMLP
from utils.numpy_compat import trapz_xy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 8 global baseline comparisons + calibration robustness.")
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--rejector-checkpoint", type=Path, required=True)
    p.add_argument("--target-coverage", type=float, default=0.8)
    p.add_argument("--calib-bootstrap", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week8_global_baselines"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_ids(split_payload: dict[str, Any], family: str) -> list[str]:
    if family == "id":
        return list(split_payload.get("test_case_ids", []))
    return list(split_payload.get("test_shift_case_ids", []) or split_payload.get("test_case_ids", []))


def evaluate_score(scores: np.ndarray, risks: np.ndarray, tau: float) -> dict[str, float]:
    accepted = scores <= tau
    coverage = float(np.mean(accepted))
    selective_risk = float(np.mean(risks[accepted])) if np.any(accepted) else float("inf")
    overall = float(np.mean(risks))
    taus, covs, rsks = risk_coverage_curve(scores, risks, num_points=60)
    m = np.isfinite(rsks)
    aurc = trapz_xy(rsks[m], covs[m]) if np.any(m) else float("inf")
    return {
        "coverage": coverage,
        "selective_risk": selective_risk,
        "overall_risk": overall,
        "aurc": aurc,
    }


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

    # Split references
    id_split = read_json(Path(manifest["split_files"]["id"]))
    val_ids = set(id_split.get("val_case_ids", []))
    val_df = df[df["case_id"].isin(val_ids)].copy()

    # Joint score from trained rejector
    model = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1).to(args.device)
    ckpt = torch.load(args.rejector_checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        joint_scores_val = model(
            torch.from_numpy(val_df[FEATURE_COLS].to_numpy(np.float32, copy=True)).to(args.device)
        ).cpu().numpy()

    score_defs = {
        "uncertainty_only": ("uncertainty_mean", 1.0),
        "residual_only": ("residual_mean", 1.0),
        "drift_only": ("drift_mean", 1.0),
        "ood_only": ("ood_centroid_distance", 1.0),
        # Lower is safer, same convention as rejector score.
    }

    # Calibrate taus on ID validation for matched coverage.
    tau_map: dict[str, float] = {}
    for name, (col, sign) in score_defs.items():
        s = sign * val_df[col].to_numpy(np.float32)
        tau_map[name] = find_threshold_for_target_coverage(s, args.target_coverage)
    tau_map["joint"] = find_threshold_for_target_coverage(joint_scores_val, args.target_coverage)

    # Calibration robustness via bootstraps on ID validation
    rng = np.random.default_rng(42)
    robust_rows = []
    for method in list(score_defs.keys()) + ["joint"]:
        taus = []
        covs = []
        risks = []
        for _ in range(args.calib_bootstrap):
            idx = rng.choice(len(val_df), size=max(2, len(val_df)), replace=True)
            val_boot = val_df.iloc[idx]
            if method == "joint":
                with torch.no_grad():
                    sb = model(
                        torch.from_numpy(val_boot[FEATURE_COLS].to_numpy(np.float32, copy=True)).to(args.device)
                    ).cpu().numpy()
            else:
                col, sign = score_defs[method]
                sb = sign * val_boot[col].to_numpy(np.float32)
            rb = val_boot["rollout_rmse"].to_numpy(np.float32)
            tau_b = find_threshold_for_target_coverage(sb, args.target_coverage)
            em = evaluate_score(sb, rb, tau_b)
            taus.append(tau_b)
            covs.append(em["coverage"])
            risks.append(em["selective_risk"])
        robust_rows.append(
            {
                "method": method,
                "tau_mean": float(np.mean(taus)),
                "tau_std": float(np.std(taus)),
                "coverage_mean": float(np.mean(covs)),
                "coverage_std": float(np.std(covs)),
                "selective_risk_mean": float(np.mean(risks)),
                "selective_risk_std": float(np.std(risks)),
            }
        )

    # Evaluate all methods on all split families.
    family_map = {
        "id": manifest["split_files"]["id"],
        "parameter_shift": manifest["split_files"]["parameter_shift"],
        "geometry_shift": manifest["split_files"]["geometry_shift"],
        "long_rollout": manifest["split_files"]["long_rollout"],
    }
    rows = []
    for fam, split_path_str in family_map.items():
        sp = read_json(Path(split_path_str))
        ids = get_ids(sp, fam)
        sdf = df[df["case_id"].isin(ids)].copy()
        if sdf.empty:
            continue
        risk = sdf["rollout_rmse"].to_numpy(np.float32)

        # single-signal methods
        for method, (col, sign) in score_defs.items():
            scores = sign * sdf[col].to_numpy(np.float32)
            ev = evaluate_score(scores, risk, tau_map[method])
            rows.append({"family": fam, "method": method, "tau": tau_map[method], "num_cases": len(sdf), **ev})

        # joint rejector
        with torch.no_grad():
            js = model(
                torch.from_numpy(sdf[FEATURE_COLS].to_numpy(np.float32, copy=True)).to(args.device)
            ).cpu().numpy()
        evj = evaluate_score(js, risk, tau_map["joint"])
        rows.append({"family": fam, "method": "joint", "tau": tau_map["joint"], "num_cases": len(sdf), **evj})

        # plot risk-coverage for this family
        fig = plt.figure(figsize=(7, 4))
        for method, (col, sign) in score_defs.items():
            ss = sign * sdf[col].to_numpy(np.float32)
            _, cov, rsk = risk_coverage_curve(ss, risk, num_points=60)
            plt.plot(cov, rsk, label=method)
        _, cov, rsk = risk_coverage_curve(js, risk, num_points=60)
        plt.plot(cov, rsk, label="joint", linewidth=2.0)
        plt.xlabel("Coverage")
        plt.ylabel("Selective risk (rollout RMSE)")
        plt.title(f"Week8 Risk-Coverage Comparison: {fam}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"risk_coverage_compare_{fam}.png", dpi=180)
        plt.close(fig)

    write_csv(args.output_dir / "week8_global_baseline_comparison.csv", rows)
    write_csv(args.output_dir / "week8_calibration_robustness.csv", robust_rows)
    (args.output_dir / "week8_global_baseline_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "week8_calibration_robustness.json").write_text(
        json.dumps(robust_rows, indent=2), encoding="utf-8"
    )

    # Summary markdown
    card = [
        "# Week 8 Global Baseline Comparison",
        "",
        f"- Target coverage: `{args.target_coverage}`",
        "",
        "## Calibrated thresholds (ID validation)",
    ]
    for k, v in tau_map.items():
        card.append(f"- `{k}`: `{v:.6f}`")
    card.extend(["", "## Calibration robustness (bootstrap on ID val)"])
    for rr in robust_rows:
        card.append(
            f"- `{rr['method']}`: tau_std={rr['tau_std']:.6f}, "
            f"coverage_std={rr['coverage_std']:.4f}, selective_risk_std={rr['selective_risk_std']:.5f}"
        )
    (args.output_dir / "week8_summary_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[OK] Week8 comparison outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
