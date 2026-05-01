"""Periodic minimum-image distance and simple cell binning."""

from __future__ import annotations

import math


def cround(x):
    return math.ceil(x - 0.5) if x < 0.0 else math.floor(x + 0.5)


def min_img_dist_sq(x, y, box, periodic=True):
    dist = 0.0
    for i in range(3):
        dx = x[i] - y[i]
        if periodic:
            dx -= cround(dx / box[i]) * box[i]
        dist += dx * dx
    return dist


def bin_particles(positions, box, cell_number):
    n_cells = cell_number[0] * cell_number[1] * cell_number[2]
    head = [-1] * n_cells
    cells = [-1] * len(positions)
    for i, pos in enumerate(positions):
        icell = 0
        for j in range(3):
            k = (pos[j] / box[j]) * cell_number[j]
            k = math.floor(k % cell_number[j])
            icell = int(k) + icell * cell_number[j]
        cells[i] = head[icell]
        head[icell] = i
    return head, cells
