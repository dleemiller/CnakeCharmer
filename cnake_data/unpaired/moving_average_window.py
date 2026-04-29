"""Rolling moving-average kernel with O(n) incremental updates."""

from __future__ import annotations

import numpy as np


def moving_average(x, m):
    if m <= 0:
        raise ValueError("window size must be positive")

    n = len(x)
    x_ma = np.full(n, np.nan)
    next_value = 0.0

    for i in range(n):
        next_value += x[i] / m
        if i >= m - 1:
            x_ma[i] = next_value
            next_value -= x[i - m + 1] / m

    return x_ma
