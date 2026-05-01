from __future__ import annotations


def biharmonic_step(z: list[list[float]], alpha: float) -> list[list[float]]:
    h = len(z)
    if h == 0:
        return []
    w = len(z[0])
    out = [row[:] for row in z]
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            lap = z[y - 1][x] + z[y + 1][x] + z[y][x - 1] + z[y][x + 1] - 4.0 * z[y][x]
            lap2 = z[y - 2][x] + z[y + 2][x] + z[y][x - 2] + z[y][x + 2] - 4.0 * z[y][x]
            out[y][x] = z[y][x] + alpha * (lap2 - 2.0 * lap)
    return out
