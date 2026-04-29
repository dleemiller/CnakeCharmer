from __future__ import annotations

import math


def cosine_similarity(u: list[float], v: list[float]) -> float:
    if len(u) != len(v):
        raise ValueError("length mismatch")
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for i in range(len(u)):
        a = u[i]
        b = v[i]
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / math.sqrt(nu * nv)


def all_pairs_cosine(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
        for j in range(i + 1, n):
            c = cosine_similarity(vectors[i], vectors[j])
            out[i][j] = c
            out[j][i] = c
    return out
