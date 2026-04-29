"""Estimate pi using coprime pair counting."""

from __future__ import annotations

import math


def calc_pi(n):
    s = 0
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if math.gcd(a, b) == 1:
                s += 1
    return math.sqrt(6.0 * (n**2) / s)
