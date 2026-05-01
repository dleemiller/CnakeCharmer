"""Roche-potential and effective-gravity spherical helpers."""

from __future__ import annotations

import math


def r(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def rp(x, y, z):
    return math.sqrt((1 - x) * (1 - x) + y * y + z * z)


def roche(x, y, z, q):
    if q == 0:
        return -(r(x, y, z) ** -1.0) - 0.5 * (x * x + y * y)
    return -(r(x, y, z) ** -1.0) - q * ((rp(x, y, z) ** -1.0) - x) - 0.5 * (1 + q) * (x * x + y * y)


def drochedrspherical(ar, sint, cosp, q):
    if q == 0:
        return ar**-2.0 - ar * sint * sint
    rp2 = ar * ar - 2 * ar * cosp * sint + 1
    return (
        ar**-2.0 + q * ((rp2**-1.5) * (ar - cosp * sint) + cosp * sint) - (1 + q) * ar * sint * sint
    )


def gravityspherical(ar, cost, sint, cosp, sinp, q):
    dr = drochedrspherical(ar, sint, cosp, q)
    if q == 0:
        dt = -(1 + q) * ar * ar * sint * cost
        return math.sqrt(dr * dr + (dt * dt) / (ar * ar))
    rp2 = ar * ar - 2 * ar * cosp * sint + 1
    dt = -q * ar * cosp * cost * (rp2**-1.5 - 1) - (1 + q) * ar * ar * sint * cost
    dp = q * sinp * (1 - rp2**-1.5)
    return math.sqrt(dr * dr + (dt * dt) / (ar * ar) + dp * dp)
