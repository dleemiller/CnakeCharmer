"""LDL decomposition for symmetric matrices."""

from __future__ import annotations

import numpy as np


def ldl(a):
    a = np.asarray(a, dtype=float)
    if a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")

    n = a.shape[0]
    l = np.zeros_like(a)
    d = np.zeros_like(a)

    for i in range(n):
        for j in range(n):
            if i == j:
                d[i, i] = a[i, i]
                for k in range(i):
                    d[i, i] -= (l[i, k] ** 2) * d[k, k]
                l[i, i] = 1.0
            elif j <= i:
                l[i, j] = a[i, j]
                for k in range(j):
                    l[i, j] -= l[i, k] * d[k, k] * l[j, k]
                l[i, j] *= 1.0 / d[j, j]

    return l, d
