from __future__ import annotations


def integrate_right(f, a: float, b: float, n: int) -> float:
    step = (b - a) / n
    s = 0.0
    for x in range(1, n + 1):
        s += f(a + step * x) * step
    return s


def integrate_midpoint(f, a: float, b: float, n: int) -> float:
    step = (b - a) / n
    mid = step / 2.0
    s = 0.0
    for x in range(1, n + 1):
        s += f(a + step * x - mid) * step
    return s
