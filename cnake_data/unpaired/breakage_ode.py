"""Breakage ODE right-hand-side and moment helpers."""

from __future__ import annotations

import numpy as np


def breakage_rates(number, brk_mat, slc_vec):
    """Compute dN/dt for a simple triangular breakage system."""
    n = len(number)
    dndt = np.zeros(n, dtype=float)

    total = 0.0
    for j in range(n):
        total += brk_mat[0, j] * slc_vec[j] * number[j]
    dndt[0] = total

    for i in range(1, n):
        total = 0.0
        for j in range(i, n):
            total += brk_mat[i, j] * slc_vec[j] * number[j]
        dndt[i] = total - slc_vec[i] * number[i]

    return dndt


def breakage_moment(y, brk_mat, slc_vec, lengths):
    """Append 0th-3rd moments to the breakage derivative vector."""
    n = len(y) - 4
    number = y[:n]
    dndt = breakage_rates(number, brk_mat, slc_vec)

    m0 = np.sum(dndt)
    m1 = np.sum(lengths @ dndt)
    m2 = np.sum(np.power(lengths, 2) @ dndt)
    m3 = np.sum(np.power(lengths, 3) @ dndt)

    return np.append(dndt, [m0, m1, m2, m3])
