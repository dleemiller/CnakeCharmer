"""2D heat equation FTCS solver with periodic boundaries."""

from __future__ import annotations

import numpy as np


def loop_ftcs(
    nx: int, ny: int, dx: float, dy: float, dt: float, ft: float, alpha: float, u: np.ndarray
) -> np.ndarray:
    nt = int(ft / dt)
    const_mult = alpha * dt / (dx * dx)
    u = np.asarray(u, dtype=float)
    utemp = np.zeros((nx + 2, ny + 2), dtype=float)

    t = 0
    while t < nt:
        t += 1
        utemp[:, :] = u[:, :]
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                u[i, j] = utemp[i, j] + const_mult * (
                    -4.0 * utemp[i, j]
                    + (utemp[i - 1, j] + utemp[i + 1, j] + utemp[i, j - 1] + utemp[i, j + 1])
                )
        for j in range(1, ny + 1):
            u[0, j] = u[nx, j]
            u[nx + 1, j] = u[1, j]
        for i in range(0, nx + 2):
            u[i, 0] = u[i, ny]
            u[i, ny + 1] = u[i, 1]
    return u
