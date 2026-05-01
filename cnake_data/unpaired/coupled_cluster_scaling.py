"""Coupled-cluster amplitude scaling for dense and sparse two-body amplitudes."""

from __future__ import annotations

import numpy as np


def amplitude_scaling_two_body(
    t: np.ndarray, h: np.ndarray, m: int, n: int, tol: float = 1e-10
) -> None:
    for a in range(m):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    divisor = h[i, i] + h[j, j] - h[a + n, a + n] - h[b + n, b + n]
                    if abs(divisor) < tol:
                        continue
                    val = t[a, b, i, j] / divisor
                    t[a, b, i, j] = val
                    t[a, b, j, i] = -val
                    t[b, a, i, j] = -val
                    t[b, a, j, i] = val


def amplitude_scaling_two_body_sparse(
    indices: np.ndarray, data: np.ndarray, h: np.ndarray, n: int, tol: float = 1e-10
) -> None:
    a_arr = indices[0]
    b_arr = indices[1]
    i_arr = indices[2]
    j_arr = indices[3]
    for index in range(len(data)):
        a = int(a_arr[index])
        b = int(b_arr[index])
        i = int(i_arr[index])
        j = int(j_arr[index])
        divisor = h[i, i] + h[j, j] - h[n + a, n + a] - h[n + b, n + b]
        if abs(divisor) < tol:
            continue
        data[index] = data[index] / divisor
