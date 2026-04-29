"""Stirling numbers of the second kind."""

from __future__ import annotations

import math


def stirling_s2(n, k):
    if k > n:
        return 0
    if n == k:
        return 1
    if k == 0:
        return 0
    if k == 1:
        return 1
    if k + 1 == n:
        return n * (n - 1) // 2
    if k == 2:
        return (1 << (n - 1)) - 1

    s = 0
    for j in range(0, k + 1):
        s += ((-1) ** (k - j)) * math.comb(k, j) * (j**n)
    return s // math.factorial(k)
