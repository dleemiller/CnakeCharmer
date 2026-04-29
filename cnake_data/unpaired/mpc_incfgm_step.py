from __future__ import annotations


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    out = [0.0] * len(a)
    for i, row in enumerate(a):
        s = 0.0
        for j, v in enumerate(row):
            s += v * x[j]
        out[i] = s
    return out


def grad_step(h: list[list[float]], g: list[float], u: list[float], alpha: float) -> list[float]:
    hu = matvec(h, u)
    out = [0.0] * len(u)
    for i in range(len(u)):
        out[i] = u[i] - alpha * (hu[i] + g[i])
    return out
