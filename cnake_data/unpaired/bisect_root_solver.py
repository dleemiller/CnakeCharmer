"""Bisection root solver for quartic family."""

from __future__ import annotations

import numpy as np


def root_func(x: float, aa: float, kk: float):
    return x**4 + kk * x**2 + aa * x - 3


def python_bisect(a: float, b: float, aa: float, kk: float, tol: float, mxiter: int):
    its = 0
    fa = root_func(a, aa, kk)
    fb = root_func(b, aa, kk)
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    c = (a + b) / 2.0
    fc = root_func(c, aa, kk)
    while abs(fc) > tol and its < mxiter:
        its += 1
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        c = (a + b) / 2.0
        fc = root_func(c, aa, kk)
    return c


def main(a: np.ndarray, kk: np.ndarray):
    l_arr = a.shape[0]
    res = np.zeros(l_arr, dtype=np.double)
    for i in range(l_arr):
        res[i] = python_bisect(0.0, 2.0, float(a[i]), float(kk[i]), 2e-12, 100)
    return res
