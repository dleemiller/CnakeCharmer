"""Dynamic time warping distance kernels."""

from __future__ import annotations

import numpy as np


def _min3(a: float, b: float, c: float) -> float:
    m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


def dtw_dist(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = x.shape[0]
    c = y.shape[0]
    D = np.zeros((r + 1, c + 1), dtype=np.float64)
    D[1:, 0] = np.inf
    D[0, 1:] = np.inf

    for i in range(r):
        for j in range(c):
            cost = float(np.linalg.norm(x[i] - y[j], ord=1))
            D[i + 1, j + 1] = cost + _min3(D[i, j + 1], D[i + 1, j], D[i, j])

    return float(D[r, c] / (r + c))


def local_dtw(s: np.ndarray, t: np.ndarray, window: int = 4) -> tuple[float, np.ndarray]:
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    r = s.shape[0]
    c = t.shape[0]
    D = np.inf * np.ones((r + 1, c + 1), dtype=np.float64)
    window = max(window, abs(r - c))
    D[0, 0] = 0.0

    for i in range(r):
        for j in range(max(0, i - window), min(c, i + window + 1)):
            cost = float(np.linalg.norm(s[i] - t[j], ord=1))
            D[i + 1, j + 1] = cost + _min3(D[i, j + 1], D[i + 1, j], D[i, j])

    return float(D[r, c] / (r + c)), D[1:, 1:]
