from __future__ import annotations


def integrate_trapezoid(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x/y length mismatch")
    if len(x) < 2:
        return 0.0
    s = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        s += 0.5 * (y[i + 1] + y[i]) * dx
    return s
