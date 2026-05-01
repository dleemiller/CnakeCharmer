"""Gaussian profile construction for one-dimensional sampled data."""

from __future__ import annotations

import math


def gauss_1d(data: list[float], std: float) -> list[float]:
    """Return normalized Gaussian weights centered on data midpoint."""
    n = len(data)
    if n == 0:
        return []

    cent = 0.5 * (n - 1)
    inv = 1.0 / (2.0 * std * std)
    norm = 1.0 / (std * math.sqrt(2.0 * math.pi))

    out = [0.0] * n
    for i in range(n):
        x = i - cent
        out[i] = norm * math.exp(-(x * x) * inv)

    s = sum(out)
    if s > 0:
        out = [v / s for v in out]
    return out


def gauss_obp(signal: list[float], std: float) -> list[float]:
    """Apply Gaussian weighting profile to signal."""
    kernel = gauss_1d(signal, std)
    return [v * w for v, w in zip(signal, kernel, strict=False)]
