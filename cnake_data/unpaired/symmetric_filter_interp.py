"""Symmetric exponential filter and 5th-order interpolation stencil."""

from __future__ import annotations

import numpy as np


def sym_exp_filt(X, X_len: int, C0: float, z: float, K0: int, KVec):
    out = np.empty(X_len, dtype=float)
    out[0] = 0.0
    for k in range(K0):
        out[0] += (z**k) * X[KVec[k]]
    for k in range(1, X_len):
        out[k] = X[k] + z * out[k - 1]
    out[X_len - 1] = (2 * out[X_len - 1] - X[X_len - 1]) * C0
    for k in range(X_len - 2, -1, -1):
        out[k] = (out[k + 1] - out[k]) * z
    return out


def bs5_int(IA, n_rows: int, n_cols: int, new_x, new_y):
    IA = np.asarray(IA, dtype=float)
    new_x = np.asarray(new_x, dtype=float)
    new_y = np.asarray(new_y, dtype=float)
    IB = np.empty((n_rows, n_cols), dtype=float)
    c4 = 1.0 / 120.0

    for i in range(n_rows):
        for j in range(n_cols):
            xn = int(new_x[i, j])
            yn = int(new_y[i, j])
            rad1, rad2 = xn - 2, xn + 3
            rad3, rad4 = yn - 2, yn + 3

            w = new_x[i, j] - xn
            w2, w3, w4, w5 = w * w, w * w * w, w**4, w**5
            wx = np.array(
                [
                    c4 * (1 - 5 * w + 10 * w2 - 10 * w3 + 5 * w4 - w5),
                    c4 * (26 - 50 * w + 20 * w2 + 20 * w3 - 20 * w4 + 5 * w5),
                    c4 * (66 - 60 * w2 + 30 * w4 - 10 * w5),
                    c4 * (26 + 50 * w + 20 * w2 - 20 * w3 - 20 * w4 + 10 * w5),
                    c4 * (1 + 5 * w + 10 * w2 + 10 * w3 + 5 * w4 - 5 * w5),
                    c4 * w5,
                ]
            )

            w = new_y[i, j] - yn
            w2, w3, w4, w5 = w * w, w * w * w, w**4, w**5
            wy = np.array(
                [
                    c4 * (1 - 5 * w + 10 * w2 - 10 * w3 + 5 * w4 - w5),
                    c4 * (26 - 50 * w + 20 * w2 + 20 * w3 - 20 * w4 + 5 * w5),
                    c4 * (66 - 60 * w2 + 30 * w4 - 10 * w5),
                    c4 * (26 + 50 * w + 20 * w2 - 20 * w3 - 20 * w4 + 10 * w5),
                    c4 * (1 + 5 * w + 10 * w2 + 10 * w3 + 5 * w4 - 5 * w5),
                    c4 * w5,
                ]
            )

            bf_a = 0.0
            for cy, ii in enumerate(range(rad3 - 1, rad4)):
                ri = ii
                if ri < 0:
                    ri = -ri
                if ri > n_rows - 1:
                    ri = 2 * (n_rows - 1) - ri
                for cx, jj in enumerate(range(rad1 - 1, rad2)):
                    rj = jj
                    if rj < 0:
                        rj = -rj
                    if rj > n_cols - 1:
                        rj = 2 * (n_cols - 1) - rj
                    bf_a += wx[cx] * wy[cy] * IA[ri, rj]
            IB[i, j] = bf_a
    return IB
