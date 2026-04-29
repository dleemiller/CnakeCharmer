from __future__ import annotations

import math


def gram_schmidt(vectors: list[list[float]], eps: float = 1e-12) -> list[list[float]]:
    basis: list[list[float]] = []
    for v in vectors:
        u = v[:]
        for b in basis:
            dot = 0.0
            for i in range(len(u)):
                dot += u[i] * b[i]
            for i in range(len(u)):
                u[i] -= dot * b[i]
        n2 = 0.0
        for a in u:
            n2 += a * a
        n = math.sqrt(n2)
        if n > eps:
            basis.append([a / n for a in u])
    return basis
