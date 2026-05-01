"""Triple-loop dense matrix multiplication baseline."""

from __future__ import annotations


def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    if n == 0:
        return []
    m = len(b[0])
    kdim = len(b)
    c = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(kdim):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c
