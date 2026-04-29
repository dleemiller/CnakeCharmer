"""Explicit finite-difference heat equation evolution."""

from __future__ import annotations


def evolve(u, u_previous, a, dt, dx2, dy2):
    n = len(u)
    m = len(u[0])
    dx2inv = 1.0 / dx2
    dy2inv = 1.0 / dy2
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            u[i][j] = u_previous[i][j] + a * dt * (
                (u_previous[i + 1][j] - 2 * u_previous[i][j] + u_previous[i - 1][j]) * dx2inv
                + (u_previous[i][j + 1] - 2 * u_previous[i][j] + u_previous[i][j - 1]) * dy2inv
            )
    for i in range(n):
        for j in range(m):
            u_previous[i][j] = u[i][j]


def iterate(field, field0, a, dx, dy, timesteps):
    dx2 = dx * dx
    dy2 = dy * dy
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2))
    for _ in range(timesteps):
        evolve(field, field0, a, dt, dx2, dy2)
    return field
