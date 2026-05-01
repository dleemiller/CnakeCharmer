from __future__ import annotations

import math


def circle_circle_penetration(
    ax: float, ay: float, ar: float, bx: float, by: float, br: float
) -> float:
    d = math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))
    return ar + br - d


def circle_line_intersection_t(
    cx: float, cy: float, cr: float, lx: float, ly: float, half_len: float, rot: float
) -> float:
    ex = lx - half_len * math.cos(rot)
    ey = ly - half_len * math.sin(rot)
    lx2 = ex + 2.0 * half_len * math.cos(rot)
    ly2 = ey + 2.0 * half_len * math.sin(rot)

    dx = lx2 - ex
    dy = ly2 - ey
    fx = ex - cx
    fy = ey - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - cr * cr
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return -1.0
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2.0 * a)
    t2 = (-b + disc) / (2.0 * a)
    if 0.0 <= t1 <= 1.0:
        return t1
    if 0.0 <= t2 <= 1.0:
        return t2
    return -1.0
