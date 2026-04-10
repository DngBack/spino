from __future__ import annotations

import numpy as np


def trapz_xy(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration; works on NumPy 1.x (trapz) and 2.x (trapezoid)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]
