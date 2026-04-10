#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_batch_sampler import ResolutionBatchSampler
from datasets.local_rejector_dataset import LocalRejectorDataset
from losses.mask_regularizers import total_variation_2d
from losses.physics_loss import pde_residual_magnitude_map
from models.backbones.fno import FNO2d
from models.heads.local_rejector import LocalRejectorCNN
from utils.local_reject_targets import build_patch_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train patchwise local rejector (Week 9).")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--fno-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week9_local_rejector"))
    p.add_argument("--patch-stride", type=int, default=4, choices=[4, 8])
    p.add_argument("--risk-quantile", type=float, default=0.75, help="Per-sample quantile on patch error for unsafe label.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-tv", type=float, default=0.05)
    p.add_argument("--pos-weight", type=float, default=3.0, help="BCE positive weight for unsafe patches.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    fno: torch.nn.Module,
    loc: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
    patch: int,
    q: float,
    lambda_tv: float,
    pos_weight: float,
) -> float:
    loc.train()
    fno.eval()
    total = 0.0
    n_batch = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)
        params = batch["params"].to(device)
        dt = batch["dt"].to(device)
        with torch.no_grad():
            pred = fno(x)
        res_mag = pde_residual_magnitude_map(x, pred, params, dt, m)
        inp = torch.cat([x, pred, res_mag], dim=1)
        logits = loc(inp)
        tgt = build_patch_targets(pred, y, m, patch, q)
        if logits.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt, size=logits.shape[-2:], mode="nearest")
        pw = torch.tensor([pos_weight], device=device)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
        probs = torch.sigmoid(logits)
        tv = total_variation_2d(probs)
        loss = bce + lambda_tv * tv
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loc.parameters(), 1.0)
        optim.step()
        total += float(loss.item())
        n_batch += 1
    return total / max(n_batch, 1)


@torch.no_grad()
def eval_epoch(
    fno: torch.nn.Module,
    loc: torch.nn.Module,
    loader: DataLoader,
    device: str,
    patch: int,
    q: float,
) -> dict[str, float]:
    loc.eval()
    fno.eval()
    ious = []
    dices = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)
        params = batch["params"].to(device)
        dt = batch["dt"].to(device)
        pred = fno(x)
        res_mag = pde_residual_magnitude_map(x, pred, params, dt, m)
        inp = torch.cat([x, pred, res_mag], dim=1)
        logits = loc(inp)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()
        tgt = build_patch_targets(pred, y, m, patch, q)
        if logits.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt, size=logits.shape[-2:], mode="nearest")
        inter = (pred_mask * tgt).sum(dim=(1, 2, 3))
        union = ((pred_mask + tgt) > 0).float().sum(dim=(1, 2, 3))
        iou = (inter / (union + 1e-8)).mean().item()
        dice = (2 * inter / (pred_mask.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1e-8)).mean().item()
        ious.append(iou)
        dices.append(dice)
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    fno = FNO2d(in_channels=2, out_channels=2, width=24, depth=3, modes_h=12, modes_w=12).to(args.device)
    ck = torch.load(args.fno_checkpoint, map_location=args.device)
    fno.load_state_dict(ck["model_state_dict"])
    fno.eval()
    for p in fno.parameters():
        p.requires_grad = False

    loc = LocalRejectorCNN(in_channels=5, hidden=64, patch_stride=args.patch_stride).to(args.device)
    optim = torch.optim.AdamW(loc.parameters(), lr=args.lr, weight_decay=1e-5)

    train_ds = LocalRejectorDataset(args.manifest, args.split, split_name="train")
    val_ds = LocalRejectorDataset(args.manifest, args.split, split_name="val")
    train_bs = ResolutionBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, seed=args.seed
    )
    val_bs = ResolutionBatchSampler(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, seed=args.seed + 1
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_bs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=val_bs, num_workers=0)

    history = {"epoch": [], "train_loss": [], "val_iou": [], "val_dice": []}
    best_iou = -1.0
    for ep in range(1, args.epochs + 1):
        tl = train_epoch(
            fno,
            loc,
            train_loader,
            optim,
            args.device,
            args.patch_stride,
            args.risk_quantile,
            args.lambda_tv,
            args.pos_weight,
        )
        ev = eval_epoch(fno, loc, val_loader, args.device, args.patch_stride, args.risk_quantile)
        history["epoch"].append(ep)
        history["train_loss"].append(tl)
        history["val_iou"].append(ev["iou"])
        history["val_dice"].append(ev["dice"])
        print(f"[Epoch {ep:03d}] train_loss={tl:.4f} val_iou={ev['iou']:.4f} val_dice={ev['dice']:.4f}")
        if ev["iou"] > best_iou:
            best_iou = ev["iou"]
            torch.save({"model_state_dict": loc.state_dict(), "epoch": ep}, run_dir / "best_local_rejector.pt")

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax2 = ax.twinx()
    ax2.plot(history["epoch"], history["val_iou"], label="val_iou", color="tab:orange")
    ax2.plot(history["epoch"], history["val_dice"], label="val_dice", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax2.set_ylabel("IoU / Dice")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "local_rejector_curves.png", dpi=180)
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "fno_checkpoint": str(args.fno_checkpoint),
        "patch_stride": args.patch_stride,
        "risk_quantile": args.risk_quantile,
        "lambda_tv": args.lambda_tv,
        "best_val_iou": best_iou,
        "best_checkpoint": str(run_dir / "best_local_rejector.pt"),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Week9 local rejector training done: {run_dir}")


if __name__ == "__main__":
    main()
