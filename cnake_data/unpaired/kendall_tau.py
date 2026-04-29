"""Kendall tau-a and tau-b for ordinal vectors."""

from __future__ import annotations

from math import sqrt


def kendalltau(x, y):
    n = len(x)
    p = q = tx = ty = 0

    for i in range(n):
        x0 = x[i]
        y0 = y[i]
        for j in range(i + 1, n):
            x1 = x[j]
            y1 = y[j]
            qq = (x1 - x0) * (y1 - y0)
            if qq > 0:
                p += 1
            elif qq == 0:
                if x1 == x0 and y0 != y1:
                    tx += 1
                elif x1 != x0 and y0 == y1:
                    ty += 1
            else:
                q += 1

    if n < 2:
        return 0.0, 0.0
    tau_a = 2.0 * float(p - q) / (n * (n - 1))
    denom = float((p + q + tx) * (p + q + ty))
    tau_b = float(p - q) / sqrt(denom) if denom > 0 else 0.0
    return tau_a, tau_b
