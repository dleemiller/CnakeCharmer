"""1D mesh basis force/potential kernels (quartic and gaussian)."""

from __future__ import annotations

import math

PI = 3.14159265


def quartic_basis(x, left_edge, inv_width):
    xn = inv_width * (x - left_edge) - 1
    if abs(xn) >= 1:
        return 0.0
    return (15.0 / 16.0) * (1.0 - xn * xn) ** 2


def quartic_int_basis(x, left_edge, inv_width):
    xn = inv_width * (x - left_edge) - 1
    if xn < -1:
        return 1.0
    if xn > 1:
        return 0.0
    return -1.0 / 16.0 * (xn - 1) ** 3 * (3 * xn * xn + 9 * xn + 8)


def gaussian_basis(x, left_edge, inv_sigma):
    xn = inv_sigma * (x - left_edge) - 0.5
    if abs(xn) >= 3.5:
        return 0.0
    return 1 / math.sqrt(2 * PI) * math.exp(-0.5 * xn * xn)


def gaussian_int_basis(x, left_edge, inv_sigma):
    xn = inv_sigma * (x - left_edge) - 0.5
    if xn < -3.5:
        return 1.0
    if xn > 3.5:
        return 0.0
    return 1 - (0.5 + 0.5 * math.erf(xn / math.sqrt(2.0)))
