from __future__ import annotations

import math


def bond_lengths(
    coords: list[tuple[float, float, float]], edges: list[tuple[int, int]]
) -> list[float]:
    out: list[float] = []
    for i, j in edges:
        xi, yi, zi = coords[i]
        xj, yj, zj = coords[j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        out.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return out


def mean_bond_length(
    coords: list[tuple[float, float, float]], edges: list[tuple[int, int]]
) -> float:
    bl = bond_lengths(coords, edges)
    return sum(bl) / len(bl) if bl else 0.0
