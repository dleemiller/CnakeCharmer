from __future__ import annotations


def lj_pair_energy(r2: float, sigma: float = 1.0, eps: float = 1.0) -> float:
    if r2 <= 0:
        return 0.0
    s2 = (sigma * sigma) / r2
    s6 = s2 * s2 * s2
    s12 = s6 * s6
    return 4.0 * eps * (s12 - s6)


def total_lj_energy(
    pos: list[tuple[float, float, float]], sigma: float = 1.0, eps: float = 1.0
) -> float:
    e = 0.0
    n = len(pos)
    for i in range(n):
        xi, yi, zi = pos[i]
        for j in range(i + 1, n):
            xj, yj, zj = pos[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            e += lj_pair_energy(dx * dx + dy * dy + dz * dz, sigma, eps)
    return e
