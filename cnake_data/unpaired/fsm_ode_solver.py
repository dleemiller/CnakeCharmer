"""Finite-dimensional ODE RHS and simple integration scaffold."""

from __future__ import annotations

import numpy as np


def f(uppers, t, y):
    n = len(y)
    rs = np.empty(n, dtype=float)
    rs_sum = 0.0

    for i in range(n):
        r = t - y[i]
        if r == 0:
            raise ZeroDivisionError("t - y[i] == 0")
        rs[i] = 1.0 / r
        rs_sum += rs[i]

    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = (uppers[i] - y[i]) * (rs_sum / (n - 1) - rs[i])
    return out


def solve(initial, uppers, bids):
    initial = np.asarray(initial, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    bids = np.asarray(bids, dtype=float)

    y = initial.copy()
    n = y.size
    out = np.zeros((bids.size, n), dtype=float)
    out[0] = y

    for k in range(1, bids.size):
        dt = bids[k] - bids[k - 1]
        dydt = f(uppers, bids[k - 1], y)
        y = y + dt * dydt
        out[k] = y
    return out
