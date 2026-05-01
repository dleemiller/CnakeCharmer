"""Vectorized-style array transform kernel over N x M grid."""

from __future__ import annotations

import math


def array_double(n, m):
    inp = [[1.0 for _ in range(m)] for _ in range(n)]
    out = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            v = math.sqrt(math.exp(-math.sqrt(inp[i][j] * (i + j))))
            out[i][j] = v * v
    return out
