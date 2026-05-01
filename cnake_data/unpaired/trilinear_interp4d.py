"""Trilinear interpolation over 4D data along feature axis."""

from __future__ import annotations

import math

import numpy as np


def trilinear_interpolate4d(data, point, out=None):
    data = np.asarray(data, dtype=float)
    point = np.asarray(point, dtype=float)
    if out is None:
        out = np.zeros(data.shape[3], dtype=float)
    if data.shape[3] != out.shape[0]:
        raise ValueError("shape mismatch")

    index = [[0, 0], [0, 0], [0, 0]]
    weight = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    for i in range(3):
        if point[i] < -0.5 or point[i] >= (data.shape[i] - 0.5):
            raise ValueError("point outside")

        flr = int(math.floor(point[i]))
        rem = point[i] - flr
        index[i][0] = flr + (1 if flr == -1 else 0)
        index[i][1] = flr + (1 if flr != (data.shape[i] - 1) else 0)
        weight[i][0] = 1 - rem
        weight[i][1] = rem

    out[:] = 0.0
    n = out.shape[0]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = weight[0][i] * weight[1][j] * weight[2][k]
                for l in range(n):
                    out[l] += w * data[index[0][i], index[1][j], index[2][k], l]
    return out
