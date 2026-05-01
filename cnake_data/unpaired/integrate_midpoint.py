"""Simple midpoint numerical integration and type-inference demo."""

from __future__ import annotations


def integrate(a: float, b: float, f):
    N = 2000
    dx = (b - a) / N
    res = 0.0
    for i in range(N):
        res += f(a + i * dx)
    return res * dx


def check_type():
    i = 1
    d = 2.0
    c = 3 + 4j
    r = i * d + c
    return r
