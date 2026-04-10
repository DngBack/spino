from __future__ import annotations

import numpy as np


def coverage_at_threshold(scores: np.ndarray, tau: float) -> float:
    return float(np.mean(scores <= tau))


def risk_at_threshold(scores: np.ndarray, risks: np.ndarray, tau: float) -> float:
    mask = scores <= tau
    if np.sum(mask) == 0:
        return float("inf")
    return float(np.mean(risks[mask]))


def find_threshold_for_target_coverage(scores: np.ndarray, target_coverage: float) -> float:
    q = float(np.quantile(scores, target_coverage))
    return q


def risk_coverage_curve(scores: np.ndarray, risks: np.ndarray, num_points: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    taus = np.linspace(np.min(scores), np.max(scores), num_points)
    cov = np.array([coverage_at_threshold(scores, t) for t in taus], dtype=np.float32)
    rsk = np.array([risk_at_threshold(scores, risks, t) for t in taus], dtype=np.float32)
    return taus.astype(np.float32), cov, rsk
