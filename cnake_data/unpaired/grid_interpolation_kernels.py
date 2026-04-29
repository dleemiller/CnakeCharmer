"""Utility, binary search, interpolation and argmax kernels."""

from __future__ import annotations

import math


def utility(riskaver, con):
    if riskaver == 1.0:
        return math.log(con)
    return con ** (1.0 - riskaver) / (1.0 - riskaver)


def fast_search_single_input(grid, val, n_grid):
    if val >= grid[n_grid - 1]:
        return n_grid - 1
    if val <= grid[0]:
        return 1

    lower = -1
    upper = n_grid
    midpt = 0
    val_midpt = 0.0

    while (upper - lower) > 1:
        midpt = (upper + lower) >> 1
        val_midpt = grid[midpt]
        if val == val_midpt:
            return midpt + 1
        if val > val_midpt:
            lower = midpt
        else:
            upper = midpt

    if val > val_midpt:
        return midpt + 1
    return midpt


def get_interpolation_weight(grid, pt, n_grid):
    i1 = fast_search_single_input(grid, pt, n_grid)
    i0 = i1 - 1
    w0 = (grid[i1] - pt) / (grid[i1] - grid[i0])
    w0 = min(max(w0, 0.0), 1.0)
    return w0, i0, i1


def interpolate(grid, pt, vals, n_grid):
    w0, i0, i1 = get_interpolation_weight(grid, pt, n_grid)
    return w0 * vals[i0] + (1.0 - w0) * vals[i1]


def cargmax(vals):
    current_argmax = 0
    current_max = vals[0]
    for i in range(1, len(vals)):
        if vals[i] > current_max:
            current_max = vals[i]
            current_argmax = i
    return current_argmax
