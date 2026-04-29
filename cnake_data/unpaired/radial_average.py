"""Radial average accumulation kernels."""

from __future__ import annotations

import numpy as np


def radial_average_fastsq(arrimg, cx, cy):
    arrimg = np.asarray(arrimg, dtype=float)
    max_r = max(arrimg.shape[0] - cx, arrimg.shape[1] - cy) ** 2
    r = np.zeros((max_r, 2), dtype=float)

    xsize, ysize = arrimg.shape
    for i in range(xsize):
        for j in range(ysize):
            d = (i - cx) ** 2 + (j - cy) ** 2
            if d < max_r:
                r[d, 0] += arrimg[i, j]
                r[d, 1] += 1

    nz = np.nonzero(r[:, 1])[0]
    r[nz, 0] = r[nz, 0] / r[nz, 1]
    return r[:, 0]


def radial_average_fast(arrimg, cx, cy):
    arrimg = np.asarray(arrimg, dtype=float)
    max_x = max(arrimg.shape[0] - cx, cx)
    max_y = max(arrimg.shape[1] - cy, cy)
    max_r = int(np.round(np.sqrt(max_x**2 + max_y**2)))

    r = np.zeros((max_r, 2), dtype=float)
    xsize, ysize = arrimg.shape

    x = np.arange(0, xsize)
    y = np.arange(0, ysize)
    xx, yy = np.meshgrid(x, y)
    d = np.round(np.sqrt((xx - cy) ** 2 + (yy - cx) ** 2)).T.astype(int)

    for i in range(xsize):
        for j in range(ysize):
            n = d[i, j]
            if 0 <= n < max_r:
                r[n, 0] += arrimg[i, j]
                r[n, 1] += 1

    nz = np.nonzero(r[:, 1])[0]
    r[nz, 0] = r[nz, 0] / r[nz, 1]
    return r[:, 0]
