"""Inverse-distance interpolation on a regular grid from scattered residues."""

from __future__ import annotations

import numpy as np


def point_residue(x, y, xpos, ypos, values, N):
    power = 2.5
    smoothing = 0
    numerator = 0.0
    denominator = 0.0

    for i in range(N):
        dist = np.sqrt((x - xpos[i]) ** 2 + (y - ypos[i]) ** 2 + smoothing * smoothing)
        if dist < 1e-11:
            return values[i]
        numerator += values[i] / np.power(dist, power)
        denominator += 1.0 / np.power(dist, power)

    return numerator / denominator if denominator != 0 else 0.0


def inverse_distance(residues, size, geotransform):
    da = np.zeros(size[0] * size[1], dtype=float)

    xpos0, ypos0, values0 = [], [], []
    for key in residues.keys():
        xpos0.append(residues[key]["x"])
        ypos0.append(residues[key]["y"])
        values0.append(residues[key]["value"])

    N = len(xpos0)
    cxpos = np.asarray(xpos0, dtype=float)
    cypos = np.asarray(ypos0, dtype=float)
    cvalues = np.asarray(values0, dtype=float)

    xsize = size[1]
    ysize = size[0]

    for j in range(ysize):
        y = geotransform[3] + j * geotransform[5]
        for i in range(xsize):
            x = geotransform[0] + i * geotransform[1]
            da[i + j * xsize] = point_residue(x, y, cxpos, cypos, cvalues, N)

    return da.reshape(size)
