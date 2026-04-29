"""Generate structured-grid or random points in a unit hypercube."""

from __future__ import annotations

import numpy as np


def _generate_grid_points(grid_size: int, dimension: int, coords: np.ndarray):
    num_points = grid_size**dimension
    for point_id in range(num_points):
        if point_id == 0:
            coords[point_id, :] = 0.0
            continue
        shift_next_dim = True
        for dim in range(dimension):
            if shift_next_dim:
                if coords[point_id - 1, dim] >= grid_size - 1:
                    coords[point_id, dim] = 0.0
                    shift_next_dim = True
                else:
                    coords[point_id, dim] = coords[point_id - 1, dim] + 1.0
                    shift_next_dim = False
            else:
                coords[point_id, dim] = coords[point_id - 1, dim]

    dx = 1.0 / (grid_size - 1.0)
    coords *= dx


def _generate_random_points(size: int, dimension: int, coords: np.ndarray):
    coords[:, :] = np.random.rand(size, dimension)


def generate_points(size, dimension=2, grid=True):
    if grid:
        num_points = size**dimension
        coords = np.zeros((num_points, dimension), dtype=float)
        _generate_grid_points(size, dimension, coords)
    else:
        num_points = size
        coords = np.zeros((num_points, dimension), dtype=float)
        _generate_random_points(num_points, dimension, coords)
    return coords
