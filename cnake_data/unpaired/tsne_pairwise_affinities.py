from __future__ import annotations

import math


def pairwise_squared_dist(x: list[list[float]]) -> list[list[float]]:
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            xj = x[j]
            s = 0.0
            for k in range(len(xi)):
                d = xi[k] - xj[k]
                s += d * d
            out[i][j] = s
            out[j][i] = s
    return out


def gaussian_affinity_row(d2_row: list[float], sigma: float) -> list[float]:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    den = 2.0 * sigma * sigma
    row = [0.0] * len(d2_row)
    z = 0.0
    for i, d2 in enumerate(d2_row):
        v = math.exp(-d2 / den)
        row[i] = v
        z += v
    if z > 0:
        for i in range(len(row)):
            row[i] /= z
    return row
