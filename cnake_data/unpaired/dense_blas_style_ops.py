from __future__ import annotations


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m = len(a)
    k = len(a[0]) if m else 0
    n = len(b[0]) if b else 0
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        ai = a[i]
        for p in range(k):
            ap = ai[p]
            bp = b[p]
            for j in range(n):
                out[i][j] += ap * bp[j]
    return out


def dense_forward(
    x: list[list[float]], w: list[list[float]], bias: list[float]
) -> list[list[float]]:
    wt = [[w[r][c] for r in range(len(w))] for c in range(len(w[0]))]
    y = matmul(x, wt)
    for i in range(len(y)):
        for j in range(len(y[i])):
            y[i][j] += bias[j]
    return y
