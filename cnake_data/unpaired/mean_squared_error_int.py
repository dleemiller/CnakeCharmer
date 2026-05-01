"""Mean squared error for integer arrays."""

from __future__ import annotations

import numpy as np


def mse(arr1, arr2) -> float:
    a1 = np.asarray(arr1).reshape(-1)
    a2 = np.asarray(arr2).reshape(-1)
    if a1.shape != a2.shape:
        raise ValueError("arr1 and arr2 must have the same shape")
    diff = a1.astype(np.int64) - a2.astype(np.int64)
    return float(np.mean(diff * diff))
