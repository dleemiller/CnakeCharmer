from __future__ import annotations


def laplacian_step(field: list[list[float]], alpha: float) -> list[list[float]]:
    h = len(field)
    if h == 0:
        return []
    w = len(field[0])
    out = [row[:] for row in field]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = field[y][x]
            lap = field[y - 1][x] + field[y + 1][x] + field[y][x - 1] + field[y][x + 1] - 4.0 * c
            out[y][x] = c + alpha * lap
    return out
