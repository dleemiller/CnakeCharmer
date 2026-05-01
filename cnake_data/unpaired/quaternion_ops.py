from __future__ import annotations

import math


def quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    x = ax * bw + ay * bz - az * by + aw * bx
    y = -ax * bz + ay * bw + az * bx + aw * by
    z = ax * by - ay * bx + az * bw + aw * bz
    w = -ax * bx - ay * by - az * bz + aw * bw
    return w, x, y, z


def quat_norm(q: tuple[float, float, float, float]) -> float:
    w, x, y, z = q
    return math.sqrt(w * w + x * x + y * y + z * z)


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = quat_norm(q)
    if n == 0.0:
        return q
    w, x, y, z = q
    return w / n, x / n, y / n, z / n
