"""Simple vector sum and naive matrix multiply kernels."""

from __future__ import annotations


def vec_sum(X):
    total = 0.0
    for i in range(len(X)):
        total += X[i]
    return total


def naive_matmul(X, Y, Z):
    n = X.shape[0]
    p = X.shape[1]
    m = Y.shape[1]
    for i in range(n):
        for j in range(m):
            Z[i, j] = 0.0
            for k in range(p):
                Z[i, j] += X[i, k] * Y[k, j]
    return Z
