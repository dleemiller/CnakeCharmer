"""Conway's game-of-life one-step update."""

from __future__ import annotations

import numpy as np


def eval_pixel(row, col, grid, next_grid):
    n, m = grid.shape
    i = max(row - 1, 0)
    j = min(n, row + 2)
    k = max(col - 1, 0)
    l = min(m, col + 2)

    sub_grid = grid[i:j, k:l]
    sum_ = int(np.sum(sub_grid)) - int(grid[row, col])

    if grid[row, col] and not (2 <= sum_ <= 3):
        next_grid[row, col] = 0
    elif (not grid[row, col]) and sum_ == 3:
        next_grid[row, col] = 1


def conway_step(grid):
    grid = np.asarray(grid, dtype=int)
    n, m = grid.shape
    next_grid = grid.copy()
    for i in range(n):
        for j in range(m):
            eval_pixel(i, j, grid, next_grid)
    return next_grid
