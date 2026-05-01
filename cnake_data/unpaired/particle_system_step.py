from __future__ import annotations


def particle_step(
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    ax: list[float],
    ay: list[float],
    dt: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if not (len(x) == len(y) == len(vx) == len(vy) == len(ax) == len(ay)):
        raise ValueError("all arrays must be same length")

    nx, ny, nvx, nvy = x[:], y[:], vx[:], vy[:]
    for i in range(len(x)):
        nvx[i] += ax[i] * dt
        nvy[i] += ay[i] * dt
        nx[i] += nvx[i] * dt
        ny[i] += nvy[i] * dt
    return nx, ny, nvx, nvy
