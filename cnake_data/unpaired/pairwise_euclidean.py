"""Pairwise Euclidean distance matrix for row vectors."""

from __future__ import annotations

import numpy as np


def euclidean_distance(x1, x2):
    d = 0.0
    N = x1.shape[0]
    for i in range(N):
        tmp = x1[i] - x2[i]
        d += tmp * tmp
    return np.sqrt(d)


def pairwise1(X, metric="euclidean"):
    if metric != "euclidean":
        raise ValueError("unrecognized metric")

    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = euclidean_distance(X[i], X[j])
            D[i, j] = dist
            D[j, i] = dist
    return D
