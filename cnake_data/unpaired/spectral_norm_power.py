"""Spectral norm via power method benchmark kernel."""

from __future__ import annotations

from math import sqrt


def A(i: int, j: int) -> float:
    return 1.0 / (((i + j) * (i + j + 1) >> 1) + i + 1)


def A_times_u(u: list[float], v: list[float]) -> None:
    n = len(u)
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A(i, j) * u[j]
        v[i] = total


def At_times_u(u: list[float], v: list[float]) -> None:
    n = len(u)
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += A(j, i) * u[j]
        v[i] = total


def B_times_u(u: list[float], out: list[float], tmp: list[float]) -> None:
    A_times_u(u, tmp)
    At_times_u(tmp, out)


def spectral_norm(n: int) -> float:
    u = [1.0] * n
    v = [0.0] * n
    tmp = [0.0] * n

    for _ in range(10):
        B_times_u(u, v, tmp)
        B_times_u(v, u, tmp)

    vbv = 0.0
    vv = 0.0
    for ue, ve in zip(u, v, strict=False):
        vbv += ue * ve
        vv += ve * ve
    return sqrt(vbv / vv)
