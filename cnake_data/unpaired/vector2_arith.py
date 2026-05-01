from __future__ import annotations

import math


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def mul_scalar(a: tuple[float, float], s: float) -> tuple[float, float]:
    return a[0] * s, a[1] * s


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def normalize(a: tuple[float, float]) -> tuple[float, float]:
    length = math.sqrt(a[0] * a[0] + a[1] * a[1])
    if length == 0.0:
        return a
    return a[0] / length, a[1] / length
