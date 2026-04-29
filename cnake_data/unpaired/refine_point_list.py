"""Voxel-local trilinear threshold refinement."""

from __future__ import annotations

import numpy as np


def _argmin(vect):
    arg = 0
    for i in range(len(vect)):
        if vect[i] < vect[arg]:
            arg = i
    return arg


def refine_point_list(voxx, voxy, voxz, xr, yr, zr, data, pnts, guess, threshold):
    n = len(voxx)
    nguess = guess.shape[1]
    residue = np.empty(nguess, dtype=float)
    vertices = np.empty(8, dtype=float)

    for cursor in range(n):
        vx = voxx[cursor]
        vy = voxy[cursor]
        vz = voxz[cursor]

        dx = xr[vx + 1] - xr[vx]
        dy = yr[vy + 1] - yr[vy]
        dz = zr[vz + 1] - zr[vz]

        for k in range(8):
            vertices[k] = data[vx + k // 4, vy + ((k // 2) % 2), vz + (k % 2)] - threshold

        for k in range(nguess):
            c00 = vertices[0] * (1 - guess[0, k]) + vertices[4] * guess[0, k]
            c01 = vertices[1] * (1 - guess[0, k]) + vertices[5] * guess[0, k]
            c10 = vertices[2] * (1 - guess[0, k]) + vertices[6] * guess[0, k]
            c11 = vertices[3] * (1 - guess[0, k]) + vertices[7] * guess[0, k]

            c0 = c00 * (1 - guess[1, k]) + c01 * guess[1, k]
            c1 = c10 * (1 - guess[1, k]) + c11 * guess[1, k]
            residue[k] = (c0 * (1 - guess[2, k]) + c1 * guess[2, k]) ** 2

        minidx = _argmin(residue)
        pnts[cursor, 0] = guess[0, minidx] * dx + xr[vx]
        pnts[cursor, 1] = guess[1, minidx] * dy + yr[vy]
        pnts[cursor, 2] = guess[2, minidx] * dz + zr[vz]

    return pnts
