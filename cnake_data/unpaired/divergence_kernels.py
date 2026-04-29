"""Finite-difference divergence and pressure-gradient kernels."""

from __future__ import annotations

import numpy as np


def div_kernel(un, vn, dn, h):
    nx, ny = un.shape
    h2 = h
    u = np.asarray(un)
    v = np.asarray(vn)
    d = np.asarray(dn)

    for i in range(1, nx):
        for j in range(1, ny):
            d[i, j] = h2 * (
                (u[i, j] + u[i, j - 1] - u[i - 1, j] - u[i - 1, j - 1])
                + (v[i, j] + v[i - 1, j] - v[i, j - 1] - v[i - 1, j - 1])
            )


def div_pressure(pi, pxi, pyi, h):
    nx, ny = pxi.shape
    h2 = 2 * h
    p = np.asarray(pi)
    px = np.asarray(pxi)
    py = np.asarray(pyi)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            px[i, j] = (p[i + 1, j] - p[i, j] + p[i + 1, j + 1] - p[i, j + 1]) / h2
            py[i, j] = (p[i, j + 1] - p[i, j] + p[i + 1, j + 1] - p[i + 1, j]) / h2
