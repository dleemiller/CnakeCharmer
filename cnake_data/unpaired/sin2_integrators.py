from __future__ import annotations

import math


def integrate_sin2_left(a: float, b: float, n: int = 2000) -> float:
    dx = (b - a) / n
    s = 0.0
    for i in range(n):
        s += math.sin(a + i * dx) ** 2
    return s * dx


def integrate_sin2_mid(a: float, b: float, n: int = 2000) -> float:
    dx = (b - a) / n
    s = 0.0
    for i in range(n):
        s += math.sin(a + (i + 0.5) * dx) ** 2
    return s * dx
