from __future__ import annotations

import math


def sphere_hit(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    center: tuple[float, float, float],
    radius: float,
    t_min: float,
    t_max: float,
) -> tuple[bool, float]:
    oc = (origin[0] - center[0], origin[1] - center[1], origin[2] - center[2])
    a = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
    b = oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2]
    c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius
    disc = b * b - a * c
    if disc < 0.0:
        return False, 0.0
    t = (-b - math.sqrt(disc)) / a
    if t_min < t < t_max:
        return True, t
    return False, 0.0
