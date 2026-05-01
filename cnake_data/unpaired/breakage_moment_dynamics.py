"""Breakage population-balance update and low-order moments."""

from __future__ import annotations

import numpy as np


def breakage(number: np.ndarray, brk_mat: np.ndarray, slc_vec: np.ndarray) -> np.ndarray:
    n = len(number)
    r1 = np.zeros(n, dtype=float)
    r2 = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(i, n):
            s += brk_mat[i, j] * slc_vec[j] * number[j]
        r1[i] = s
        r2[i] = slc_vec[i] * number[i]
    r2[0] = 0.0
    return r1 - r2


def breakage_moment(
    Y: np.ndarray, brk_mat: np.ndarray, slc_vec: np.ndarray, L: np.ndarray
) -> np.ndarray:
    n = len(Y) - 4
    number = Y[:n]
    dndt = breakage(number, brk_mat, slc_vec)
    m0 = np.sum(dndt)
    m1 = np.sum(L @ dndt)
    m2 = np.sum((L**2) @ dndt)
    m3 = np.sum((L**3) @ dndt)
    return np.append(dndt, [m0, m1, m2, m3])
