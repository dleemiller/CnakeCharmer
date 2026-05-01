from __future__ import annotations


def matmul_int(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    if not a or not b:
        return []
    n = len(a)
    m = len(a[0])
    if m != len(b):
        raise ValueError("shape mismatch")
    p = len(b[0])
    out = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for z in range(m):
                s += a[i][z] * b[z][j]
            out[i][j] = s
    return out
