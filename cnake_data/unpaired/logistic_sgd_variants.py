"""Dense and sparse SGD variants for logistic-style models."""

from __future__ import annotations

import numpy as np


def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-v))


def logistic_regression(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    M: int,
    eta0: float,
    max_iter: int,
    l2_regularization: float,
):
    for t in range(max_iter):
        lam = eta0 / (1.0 + t)
        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r, m] * theta[m]
            hx = sigmoid(wx)
            z = lam * (y[r] - hx)
            for m in range(M):
                theta[m] += z * X[r, m] - (l2_regularization * 2 * lam * theta[m])
    return theta


def sparse_logistic_regression(
    theta: np.ndarray, sparseX, y: np.ndarray, N: int, M: int, eta0: float, max_iter: int
):
    data, indices, indptr = sparseX.data, sparseX.indices, sparseX.indptr
    for t in range(max_iter):
        lam = eta0 / (1.0 + t)
        for r in range(N):
            wx = 0.0
            c = indptr[r]
            d = indptr[r + 1]
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                wx += value * theta[param]
            hx = sigmoid(wx)
            z = lam * (y[r] - hx)
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                theta[param] += z * value
    return theta
