from __future__ import annotations


def histogram1d(x: list[float], nx: int, xmin: float, xmax: float) -> list[float]:
    out = [0.0] * nx
    normx = 1.0 / (xmax - xmin)
    fnx = float(nx)
    for tx in x:
        if tx >= xmin and tx < xmax:
            ix = int((tx - xmin) * normx * fnx)
            out[ix] += 1.0
    return out


def histogram2d(
    x: list[float],
    y: list[float],
    nx: int,
    ny: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[list[float]]:
    out = [[0.0 for _ in range(ny)] for _ in range(nx)]
    normx = 1.0 / (xmax - xmin)
    normy = 1.0 / (ymax - ymin)
    n = min(len(x), len(y))
    for i in range(n):
        tx = x[i]
        ty = y[i]
        if tx >= xmin and tx < xmax and ty >= ymin and ty < ymax:
            ix = int((tx - xmin) * normx * nx)
            iy = int((ty - ymin) * normy * ny)
            out[ix][iy] += 1.0
    return out
