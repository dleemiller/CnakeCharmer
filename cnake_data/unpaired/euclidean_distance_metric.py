"""Euclidean squared distance metric."""

from __future__ import annotations


def euclidean(x, y):
    n = len(x)
    res = 0.0
    for i in range(n):
        res += (x[i] - y[i]) ** 2
    return res
