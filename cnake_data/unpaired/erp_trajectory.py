"""Edit distance with real penalty (ERP) for trajectories."""

from __future__ import annotations

import math


def eucl_dist(p1, p2):
    s = 0.0
    for a, b in zip(p1, p2, strict=False):
        d = a - b
        s += d * d
    return math.sqrt(s)


def e_erp(t0, t1, g):
    n0 = len(t0) + 1
    n1 = len(t1) + 1
    c = [[0.0] * n1 for _ in range(n0)]

    edgei = 0.0
    for i in range(1, n0):
        edgei += abs(eucl_dist(t0[i - 1], g))
        c[i][0] = edgei

    edgej = 0.0
    for j in range(1, n1):
        edgej += abs(eucl_dist(t1[j - 1], g))
        c[0][j] = edgej

    for i in range(1, n0):
        for j in range(1, n1):
            d0 = c[i - 1][j] + eucl_dist(t0[i - 1], g)
            d1 = c[i][j - 1] + eucl_dist(g, t1[j - 1])
            d01 = c[i - 1][j - 1] + eucl_dist(t0[i - 1], t1[j - 1])
            c[i][j] = min(d0, d1, d01)
    return c[n0 - 1][n1 - 1]
