#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_batch_sampler import ResolutionBatchSampler
from datasets.ep_operator_dataset import EPOneStepDataset
from models.backbones.fno import FNO2d
from trainers.operator_trainer import TrainConfig, train_operator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Week-3 FNO baseline.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week3_fno"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--modes", type=int, default=16)
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

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EPOneStepDataset(args.manifest, args.split, split_name="train")
    val_ds = EPOneStepDataset(args.manifest, args.split, split_name="val")
    test_ds = EPOneStepDataset(args.manifest, args.split, split_name="test")
    train_bs = ResolutionBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, seed=args.seed
    )
    val_bs = ResolutionBatchSampler(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, seed=args.seed + 1
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_bs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=val_bs, num_workers=0)

    model = FNO2d(
        in_channels=2,
        out_channels=2,
        width=args.width,
        depth=args.depth,
        modes_h=args.modes,
        modes_w=args.modes,
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    history = train_operator(model, train_loader, val_loader, cfg, run_dir)

    summary = {
        "run_dir": str(run_dir),
        "seed": args.seed,
        "device": args.device,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_checkpoint": str(run_dir / "best_fno.pt"),
        "final_val_rmse": history["val_rmse"][-1] if history["val_rmse"] else None,
        "final_val_mae": history["val_mae"][-1] if history["val_mae"] else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Training complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
