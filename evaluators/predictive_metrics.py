from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def relative_rmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    num = np.sqrt(np.mean((pred - target) ** 2))
    den = np.sqrt(np.mean(target**2)) + eps
    return float(num / den)
