from __future__ import annotations

import math


def pair_energy(
    charges: list[float],
    coords: list[tuple[float, float, float]],
    alpha: float,
    eps: float = 1e-12,
) -> float:
    """Simplified pairwise real-space Ewald-like energy accumulation."""
    if len(charges) != len(coords):
        raise ValueError("charges/coords length mismatch")

    e = 0.0
    n = len(charges)
    for i in range(n):
        qi = charges[i]
        xi, yi, zi = coords[i]
        for j in range(i + 1, n):
            qj = charges[j]
            xj, yj, zj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            r = math.sqrt(dx * dx + dy * dy + dz * dz) + eps
            e += qi * qj * math.erfc(alpha * r) / r
    return e
