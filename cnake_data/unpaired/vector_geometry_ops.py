"""2D vector and line geometry helpers."""

from __future__ import annotations

import math

import numpy as np


def dot(v1, v2) -> float:
    return float(v1[0] * v2[0] + v1[1] * v2[1])


def norm(vec) -> float:
    return math.sqrt(dot(vec, vec))


def normalize(vec):
    l = norm(vec)
    return np.array([vec[0] / l, vec[1] / l], dtype=float)


def vec_mag(vec, mag: float):
    out = normalize(vec)
    out[0] *= mag
    out[1] *= mag
    return out


def distance(v1, v2) -> float:
    return norm(np.asarray(v2) - np.asarray(v1))


def norm_vec(vec):
    perp = np.array([-vec[1], vec[0]], dtype=float)
    return normalize(perp)


def ccw(a, b, c) -> bool:
    x = (c[1] - a[1]) * (b[0] - a[0])
    y = (b[1] - a[1]) * (c[0] - a[0])
    return x > y


def intersect(l1s, l1e, l2s, l2e) -> bool:
    c1 = ccw(l1s, l2s, l2e) != ccw(l1e, l2s, l2e)
    c2 = ccw(l1s, l1e, l2s) != ccw(l1s, l1e, l2e)
    return c1 and c2


def intersection_point(l1s, l1e, l2s, l2e):
    x1, y1 = l1s
    x2, y2 = l1e
    x3, y3 = l2s
    x4, y4 = l2e
    a = (y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)
    b = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    t = a / b
    return np.asarray(l1s) + t * (np.asarray(l1e) - np.asarray(l1s))


def reflect(vec, base):
    n = norm_vec(base)
    d = np.asarray(vec)
    return d - 2 * dot(d, n) * n


def neighbor_indices(x: int, y: int, n: int):
    return [
        (i, j)
        for i in range(x - 1, x + 2)
        for j in range(y - 1, y + 2)
        if 0 <= i < n and 0 <= j < n
    ]
