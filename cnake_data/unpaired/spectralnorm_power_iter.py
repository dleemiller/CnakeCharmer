from __future__ import annotations


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    out = [0.0] * len(a)
    for i, row in enumerate(a):
        s = 0.0
        for j, v in enumerate(row):
            s += v * x[j]
        out[i] = s
    return out


def mat_t_vec(a: list[list[float]], x: list[float]) -> list[float]:
    m = len(a[0]) if a else 0
    out = [0.0] * m
    for i, row in enumerate(a):
        xi = x[i]
        for j, v in enumerate(row):
            out[j] += v * xi
    return out


def spectral_norm_estimate(a: list[list[float]], iters: int = 20) -> float:
    n = len(a[0]) if a else 0
    if n == 0:
        return 0.0
    v = [1.0 / n] * n
    for _ in range(iters):
        av = matvec(a, v)
        atav = mat_t_vec(a, av)
        norm = sum(x * x for x in atav) ** 0.5
        if norm == 0.0:
            return 0.0
        v = [x / norm for x in atav]
    av = matvec(a, v)
    return sum(x * x for x in av) ** 0.5
