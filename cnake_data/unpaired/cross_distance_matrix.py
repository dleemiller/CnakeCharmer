from __future__ import annotations


def cross_distance_matrix(
    a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]
) -> list[list[float]]:
    out = [[0.0] * len(b) for _ in range(len(a))]
    for i, (ax, ay, az) in enumerate(a):
        row = out[i]
        for j, (bx, by, bz) in enumerate(b):
            dx = ax - bx
            dy = ay - by
            dz = az - bz
            row[j] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return out
