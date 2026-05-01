"""Binomial coefficient computation."""

from __future__ import annotations


def choose(n: int, k: int) -> float:
    if k < 0:
        return 0.0
    if k == 0:
        return 1.0
    if n < k:
        return 0.0
    p = 1.0
    N = min(k, n - k) + 1
    for i in range(1, N):
        p *= n
        p = int(p / i)
        n -= 1
    return p
