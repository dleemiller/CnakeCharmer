from __future__ import annotations

import math


def side_lengths(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float, float]:
    def d(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return math.sqrt(dx * dx + dy * dy)

    return d(a, b), d(b, c), d(c, a)


def area(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * 0.5


def is_degenerate(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], eps: float = 1e-12
) -> bool:
    return area(a, b, c) <= eps
