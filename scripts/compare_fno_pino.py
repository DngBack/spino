#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FNO vs PINO comparison artifacts.")
    p.add_argument("--fno-metrics", type=Path, required=True)
    p.add_argument("--pino-metrics", type=Path, required=True)
    p.add_argument("--fno-history", type=Path, required=True)
    p.add_argument("--pino-history", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week4_comparison"))
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fno_m = read_json(args.fno_metrics)
    pino_m = read_json(args.pino_metrics)
    fno_h = read_json(args.fno_history)
    pino_h = read_json(args.pino_history)

    keys = [
        "one_step_rmse",
        "one_step_mae",
        "one_step_relative_rmse",
        "rollout_rmse",
        "rollout_mae",
        "rollout_relative_rmse",
        "model_steps_per_second",
    ]
    rows = []
    for k in keys:
        rows.append({"metric": k, "fno": fno_m.get(k), "pino": pino_m.get(k)})

    with (out / "fno_vs_pino_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "fno", "pino"])
        w.writeheader()
        w.writerows(rows)

    # Training curve comparison
    fig = plt.figure(figsize=(8, 4))
    plt.plot(fno_h["epoch"], fno_h["val_rmse"], label="FNO val_rmse")
    plt.plot(pino_h["epoch"], pino_h["val_rmse"], label="PINO val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE")
    plt.title("FNO vs PINO Validation RMSE")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out / "fno_vs_pino_val_rmse.png", dpi=180)
    plt.close(fig)

    # Bar plot on selected metrics (lower better, except speed)
    selected = ["one_step_rmse", "rollout_rmse", "rollout_relative_rmse"]
    x = np.arange(len(selected))
    width = 0.35
    fvals = [fno_m[k] for k in selected]
    pvals = [pino_m[k] for k in selected]
    fig = plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, fvals, width=width, label="FNO")
    plt.bar(x + width / 2, pvals, width=width, label="PINO")
    plt.xticks(x, selected, rotation=20)
    plt.ylabel("Metric value")
    plt.title("FNO vs PINO (ID test)")
    plt.legend()
    plt.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fno_vs_pino_bar.png", dpi=180)
    plt.close(fig)

    summary = {
        "fno_metrics_path": str(args.fno_metrics),
        "pino_metrics_path": str(args.pino_metrics),
        "comparison_csv": str(out / "fno_vs_pino_metrics.csv"),
        "plots": [
            str(out / "fno_vs_pino_val_rmse.png"),
            str(out / "fno_vs_pino_bar.png"),
        ],
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Wrote comparison outputs to: {out}")


if __name__ == "__main__":
    main()
