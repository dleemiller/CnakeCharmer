"""In-place Gaussian elimination (forward phase)."""

from __future__ import annotations


def gauss_elimination(M, nrows: int, ncols: int):
    for step in range(0, ncols - 1):
        pivot = M[step][step]
        for row in range(step + 1, nrows):
            ratio = M[row][step] / pivot
            M[row][step] = 0
            for col in range(step + 1, ncols):
                M[row][col] = M[row][col] - ratio * M[step][col]
    return M
