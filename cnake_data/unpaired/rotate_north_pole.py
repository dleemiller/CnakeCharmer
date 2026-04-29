"""Construct rotation matrices aligning vectors to north pole."""

from __future__ import annotations

import math


def rotate_north_pole(vectors):
    n = len(vectors)
    out = [[[0.0 for _ in range(n)] for _ in range(3)] for _ in range(3)]
    eps = 2.220446049250313e-16
    for i in range(n):
        x, y, z = vectors[i]
        mag = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / mag, y / mag, z / mag
        cosa = z
        sina = math.sqrt(max(0.0, 1.0 - cosa * cosa))
        if (1.0 - cosa) > eps and sina != 0.0:
            ax = y / sina
            ay = -x / sina
            az = 0.0
        else:
            ax = ay = az = 0.0
        vera = 1.0 - cosa

        out[0][0][i] = cosa + ax * ax * vera
        out[0][1][i] = ax * ay * vera - az * sina
        out[0][2][i] = ax * az * vera + ay * sina
        out[1][0][i] = ax * ay * vera + az * sina
        out[1][1][i] = cosa + ay * ay * vera
        out[1][2][i] = ay * az * vera - ax * sina
        out[2][0][i] = ax * az * vera - ay * sina
        out[2][1][i] = ay * az * vera + ax * sina
        out[2][2][i] = cosa + az * az * vera
    return out
