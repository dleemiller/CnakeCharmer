"""Radial distribution function computation."""

from __future__ import annotations

from math import floor, pi, sqrt

import numpy as np


def _compute_rdf(r, L, N: int, dx: float):
    inv_dx = 1.0 / dx
    x_max_sqr = (N * dx) ** 2
    result = np.zeros(N, dtype=float)

    for i in range(r.shape[0]):
        for j in range(i + 1, r.shape[0]):
            dist_sqr = 0.0
            for coord in range(3):
                dist = r[i, coord] - r[j, coord]
                if dist < -L[coord] / 2.0:
                    dist += L[coord]
                elif dist > L[coord] / 2.0:
                    dist -= L[coord]
                dist_sqr += dist * dist
            if dist_sqr <= x_max_sqr:
                idx = int(floor(sqrt(dist_sqr) * inv_dx))
                if idx < N:
                    result[idx] += 1

    k = 2.0 * pi
    for i in range(result.shape[0]):
        result[i] /= k * ((i + 0.5) * dx) ** 2
    return result


def compute_rdf(r, L, N: int, cutoff: float):
    r = np.asarray(r, dtype=float)
    L = np.asarray(L, dtype=float)
    x_max = np.min(L) / 2.0 if cutoff == -1 else cutoff
    dx = x_max / N
    return dx, _compute_rdf(r, L, N, dx)
