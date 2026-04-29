"""Local-support cubic interpolation kernels."""

from __future__ import annotations

import numpy as np


def phi(t):
    abs_t = abs(t)
    if abs_t <= 1:
        return 4 - 6 * abs_t**2 + 3 * abs_t**3
    if abs_t <= 2:
        return (2 - abs_t) ** 3
    return 0.0


def interp(x, a, b, c):
    n = len(c) - 3
    h = (b - a) / n
    l = int((x - a) / h) + 1
    m = min(l + 3, n + 3)
    pos = (x - a) / h + 2

    total = 0.0
    for ii in range(l, m + 1):
        w = phi(pos - ii)
        if w:
            total += c[ii - 1] * w
    return total


def arr_interp(x, a, b, c):
    out = np.empty(len(x), dtype=float)
    for i, xi in enumerate(x):
        out[i] = interp(xi, a, b, c)
    return out
