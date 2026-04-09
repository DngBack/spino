from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cpu"


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            mse = F.mse_loss(pred, y, reduction="sum")
            mae = F.l1_loss(pred, y, reduction="sum")
            total_mse += float(mse.item())
            total_mae += float(mae.item())
            count += int(y.numel())
    rmse = (total_mse / max(count, 1)) ** 0.5
    mae = total_mae / max(count, 1)
    return {"rmse": rmse, "mae": mae}


def train_operator(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_mae": []}
    best_val_rmse = float("inf")
    best_path = output_dir / "best_fno.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loss(model, val_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_history(history, output_dir / "training_curves.png")
    return history


def _plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_mse")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["val_rmse"], label="val_rmse")
    axes[1].plot(history["epoch"], history["val_mae"], label="val_mae")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
