"""Pairwise Euclidean distances with optional periodic wrapping."""

from __future__ import annotations

import numpy as np


def dist_wrap(x, boxl):
    bo2 = boxl / 2.0
    if x > bo2:
        return x - boxl
    return x


def pair_dist(r1, r2, boxl):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.shape[0] != r2.shape[0]:
        raise ValueError("first axis of r1 and r2 must be equal")

    imax = r1.shape[0]
    out = np.zeros(imax, dtype=float)
    for i in range(imax):
        dx = abs(r1[i, 0] - r2[i, 0])
        dy = abs(r1[i, 1] - r2[i, 1])
        dz = abs(r1[i, 2] - r2[i, 2])
        dx = dist_wrap(dx, boxl)
        dy = dist_wrap(dy, boxl)
        dz = dist_wrap(dz, boxl)
        out[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return out


def all_dist(r1, r2, boxl, wrap=True):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    imax = r1.shape[0]
    jmax = r2.shape[0]
    out = np.zeros((imax, jmax), dtype=float)
    bo2 = boxl / 2.0

    for i in range(imax):
        for j in range(jmax):
            dx = abs(r1[i, 0] - r2[j, 0])
            dy = abs(r1[i, 1] - r2[j, 1])
            dz = abs(r1[i, 2] - r2[j, 2])
            if wrap:
                if dx > bo2:
                    dx -= boxl
                if dy > bo2:
                    dy -= boxl
                if dz > bo2:
                    dz -= boxl
            out[i, j] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return out
