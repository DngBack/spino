#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import load_split_case_ids
from models.heads.global_rejector import GlobalRejectorMLP
from trainers.rejector_trainer import RejectorConfig, train_global_rejector


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
    p = argparse.ArgumentParser(description="Train global rejector from reliability features.")
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--id-split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week7_global_rejector"))
    p.add_argument("--risk-quantile", type=float, default=0.75)
    p.add_argument("--target-coverage", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    df = pd.read_csv(args.feature_csv)
    split_train = set(load_split_case_ids(args.id_split, "train"))
    split_val = set(load_split_case_ids(args.id_split, "val"))

    q = float(df["rollout_rmse"].quantile(args.risk_quantile))
    df["unsafe_label"] = (df["rollout_rmse"] >= q).astype(np.float32)

    tr = df[df["case_id"].isin(split_train)].copy()
    va = df[df["case_id"].isin(split_val)].copy()
    x_train = tr[FEATURE_COLS].to_numpy(np.float32)
    y_train_unsafe = tr["unsafe_label"].to_numpy(np.float32)
    y_train_risk = tr["rollout_rmse"].to_numpy(np.float32)
    x_val = va[FEATURE_COLS].to_numpy(np.float32)
    y_val_unsafe = va["unsafe_label"].to_numpy(np.float32)
    y_val_risk = va["rollout_rmse"].to_numpy(np.float32)

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    model = GlobalRejectorMLP(in_dim=len(FEATURE_COLS), hidden_dim=32, depth=2, dropout=0.1)
    cfg = RejectorConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        target_coverage=args.target_coverage,
        lambda_cov=0.1,
        lambda_rank=0.1,
    )
    hist = train_global_rejector(
        model=model,
        x_train=x_train,
        y_train_unsafe=y_train_unsafe,
        y_train_risk=y_train_risk,
        x_val=x_val,
        y_val_unsafe=y_val_unsafe,
        y_val_risk=y_val_risk,
        cfg=cfg,
        output_dir=run_dir,
    )
    summary = {
        "run_dir": str(run_dir),
        "feature_csv": str(args.feature_csv),
        "risk_quantile": args.risk_quantile,
        "risk_threshold_rollout_rmse": q,
        "target_coverage": args.target_coverage,
        "feature_cols": FEATURE_COLS,
        "best_checkpoint": str(run_dir / "best_global_rejector.pt"),
        "final_val_bce": hist["val_bce"][-1] if hist["val_bce"] else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Global rejector trained: {run_dir}")


if __name__ == "__main__":
    main()
