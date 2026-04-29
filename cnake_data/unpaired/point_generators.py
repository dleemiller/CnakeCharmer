"""Simple 3D point generator primitives."""

from __future__ import annotations

import math
import random


def single_point(samples):
    return [(0.0, 0.0, 0.0) for _ in range(samples)]


def disk_points(samples, radius=1.0):
    out = []
    for _ in range(samples):
        r = radius * math.sqrt(random.random())
        theta = 2.0 * math.pi * random.random()
        out.append((r * math.cos(theta), r * math.sin(theta), 0.0))
    return out
