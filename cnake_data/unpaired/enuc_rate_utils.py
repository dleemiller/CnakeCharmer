from __future__ import annotations

import math


def exp_clip(x: float, lo: float = -700.0, hi: float = 700.0) -> float:
    if x < lo:
        x = lo
    elif x > hi:
        x = hi
    return math.exp(x)


def temperature_power(t: float, p: float) -> float:
    if t <= 0:
        raise ValueError("temperature must be positive")
    return t**p
