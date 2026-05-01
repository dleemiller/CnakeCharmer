from __future__ import annotations


def square_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            out[i][j] = s
    return out
