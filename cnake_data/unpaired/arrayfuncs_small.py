"""Small array helper kernels: positive minimum and Cholesky delete."""

from __future__ import annotations

import numpy as np


def min_pos(X):
    X = np.asarray(X)
    min_val = np.finfo(X.dtype).max
    for i in range(X.size):
        if 0.0 < X[i] < min_val:
            min_val = X[i]
    return min_val


def cholesky_delete(L, go_out: int):
    A = np.array(L, copy=True)
    n = A.shape[0]
    # remove row/col then refactor
    keep = [i for i in range(n) if i != go_out]
    reduced = A[np.ix_(keep, keep)]
    # use nearest SPD fallback by symmetrization
    reduced = 0.5 * (reduced + reduced.T)
    try:
        return np.linalg.cholesky(reduced)
    except np.linalg.LinAlgError:
        eps = 1e-9
        return np.linalg.cholesky(reduced + eps * np.eye(reduced.shape[0]))
