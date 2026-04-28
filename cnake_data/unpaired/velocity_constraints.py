"""Create velocity-constraint coefficient matrices."""

from __future__ import annotations

import numpy as np

MAXSD = 1e8


def _create_velocity_constraint(qs: np.ndarray, vlim: np.ndarray):
    N = qs.shape[0] - 1
    dof = qs.shape[1]
    a = np.zeros((N + 1, 2), dtype=float)
    b = np.ones((N + 1, 2), dtype=float)
    c = np.zeros((N + 1, 2), dtype=float)
    b[:, 1] = -1

    for i in range(N + 1):
        sdmin = -MAXSD
        sdmax = MAXSD
        for k in range(dof):
            if qs[i, k] > 0:
                sdmax = min(vlim[k, 1] / qs[i, k], sdmax)
                sdmin = max(vlim[k, 0] / qs[i, k], sdmin)
            elif qs[i, k] < 0:
                sdmax = min(vlim[k, 0] / qs[i, k], sdmax)
                sdmin = max(vlim[k, 1] / qs[i, k], sdmin)
        c[i, 0] = -(sdmax**2)
        c[i, 1] = max(sdmin, 0.0) ** 2
    return a, b, c


def _create_velocity_constraint_varying(qs: np.ndarray, vlim_grid: np.ndarray):
    N = qs.shape[0] - 1
    dof = qs.shape[1]
    a = np.zeros((N + 1, 2), dtype=float)
    b = np.ones((N + 1, 2), dtype=float)
    c = np.zeros((N + 1, 2), dtype=float)
    b[:, 1] = -1

    for i in range(N + 1):
        sdmin = -MAXSD
        sdmax = MAXSD
        for k in range(dof):
            if qs[i, k] > 0:
                sdmax = min(vlim_grid[i, k, 1] / qs[i, k], sdmax)
                sdmin = max(vlim_grid[i, k, 0] / qs[i, k], sdmin)
            elif qs[i, k] < 0:
                sdmax = min(vlim_grid[i, k, 0] / qs[i, k], sdmax)
                sdmin = max(vlim_grid[i, k, 1] / qs[i, k], sdmin)
        c[i, 0] = -(sdmax**2)
        c[i, 1] = max(sdmin, 0.0) ** 2
    return a, b, c
