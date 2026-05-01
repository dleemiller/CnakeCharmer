from __future__ import annotations

import math


def sinc(x: float) -> float:
    return 1.0 if x == 0.0 else math.sin(x) / x


def sk_debye_self(
    positions: list[list[float]], k_vals: list[float], box_l: list[float]
) -> list[float]:
    n = len(positions)
    ndim = len(positions[0]) if n else 0
    sk = [1.0 for _ in k_vals]
    nk = len(k_vals)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dd = 0.0
            for p in range(ndim):
                d = positions[i][p] - positions[j][p]
                d = d - round(d / box_l[p]) * box_l[p]
                dd += d * d
            r = math.sqrt(dd)
            for m in range(nk):
                sk[m] += (2.0 / n) * sinc(k_vals[m] * r)
    return sk
