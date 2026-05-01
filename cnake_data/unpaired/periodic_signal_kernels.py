from __future__ import annotations

import math


def goertzel_power(x: list[float], period: int) -> float:
    coeff = 2.0 * math.cos(2.0 * math.pi / period)
    s_prev = 0.0
    s_prev2 = 0.0
    for xn in x:
        s = xn + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return math.sqrt(s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev2 * s_prev)


def ipdft_inner(x: list[float], w_vals: list[complex]) -> list[complex]:
    out = [0j for _ in range(len(w_vals))]
    n = len(x)
    for p in range(len(w_vals)):
        w = 1.0 + 0.0j
        for i in range(n):
            if i != 0:
                w *= w_vals[p]
            out[p] += x[i] * w
    return out


def autocorr_inner(x: list[float]) -> list[float]:
    n = len(x)
    out = [0.0 for _ in range(2 * n - 1)]
    for m in range(-n + 1, n):
        s = 0.0
        for i in range(n):
            j = i - m
            if 0 <= j < n:
                s += x[i] * x[j]
        out[m + n - 1] = s
    return out
