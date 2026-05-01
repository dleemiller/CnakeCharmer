from __future__ import annotations

import math


def _eucl(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def erp_distance(
    t0: list[tuple[float, float]], t1: list[tuple[float, float]], g: tuple[float, float]
) -> float:
    n0 = len(t0) + 1
    n1 = len(t1) + 1
    c = [[0.0 for _ in range(n1)] for _ in range(n0)]

    edge = 0.0
    for i in range(1, n0):
        edge += abs(_eucl(t0[i - 1], g))
        c[i][0] = edge

    edge = 0.0
    for j in range(1, n1):
        edge += abs(_eucl(t1[j - 1], g))
        c[0][j] = edge

    for i in range(1, n0):
        for j in range(1, n1):
            derp0 = c[i - 1][j] + _eucl(t0[i - 1], g)
            derp1 = c[i][j - 1] + _eucl(g, t1[j - 1])
            derp01 = c[i - 1][j - 1] + _eucl(t0[i - 1], t1[j - 1])
            c[i][j] = min(derp0, derp1, derp01)
    return c[n0 - 1][n1 - 1]
