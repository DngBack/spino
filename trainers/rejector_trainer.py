from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from calibration.threshold_search import find_threshold_for_target_coverage
from losses.coverage_loss import coverage_regularization
from losses.selective_loss import bce_risk_loss, pairwise_ranking_loss


@dataclass
class RejectorConfig:
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    target_coverage: float = 0.8
    lambda_cov: float = 0.1
    lambda_rank: float = 0.1


def train_global_rejector(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train_unsafe: np.ndarray,
    y_train_risk: np.ndarray,
    x_val: np.ndarray,
    y_val_unsafe: np.ndarray,
    y_val_risk: np.ndarray,
    cfg: RejectorConfig,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    model.to(device)
    ds = TensorDataset(
        torch.from_numpy(np.array(x_train, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(y_train_unsafe, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(y_train_risk, dtype=np.float32, copy=True)),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train_loss": [], "val_bce": [], "val_cov_at_tau": [], "tau": []}
    best_val = float("inf")
    best_path = output_dir / "best_global_rejector.pt"

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for xb, yb_unsafe, yb_risk in loader:
            xb = xb.to(device)
            yb_unsafe = yb_unsafe.to(device)
            yb_risk = yb_risk.to(device)
            scores = model(xb)
            tau = torch.quantile(scores.detach(), cfg.target_coverage)
            loss = bce_risk_loss(scores, yb_unsafe)
            loss = loss + cfg.lambda_rank * pairwise_ranking_loss(scores, yb_risk)
            loss = loss + cfg.lambda_cov * coverage_regularization(scores, tau, cfg.target_coverage)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            batches += 1

        model.eval()
        with torch.no_grad():
            xvt = torch.from_numpy(np.array(x_val, dtype=np.float32, copy=True)).to(device)
            yvt = torch.from_numpy(np.array(y_val_unsafe, dtype=np.float32, copy=True)).to(device)
            scores_val = model(xvt).cpu().numpy()
            val_bce = float(
                bce_risk_loss(
                    torch.from_numpy(np.asarray(scores_val, dtype=np.float32)),
                    torch.from_numpy(np.array(y_val_unsafe, dtype=np.float32, copy=True)),
                ).item()
            )
            tau = find_threshold_for_target_coverage(scores_val, cfg.target_coverage)
            val_cov = float(np.mean(scores_val <= tau))

        history["epoch"].append(ep)
        history["train_loss"].append(running / max(batches, 1))
        history["val_bce"].append(val_bce)
        history["val_cov_at_tau"].append(val_cov)
        history["tau"].append(float(tau))

        if val_bce < best_val:
            best_val = val_bce
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, best_path)

    (output_dir / "rejector_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_history(history, output_dir / "rejector_training_curves.png")
    return history


def _plot_history(hist: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train_loss")
    axes[0].set_title("Rejector Train Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(hist["epoch"], hist["val_bce"], label="val_bce")
    axes[1].plot(hist["epoch"], hist["val_cov_at_tau"], label="val_cov_at_tau")
    axes[1].set_title("Validation")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
