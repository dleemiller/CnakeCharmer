from __future__ import annotations


def nbody_accelerations(
    pos: list[tuple[float, float, float]], masses: list[float], g: float = 1.0, eps: float = 1e-9
) -> list[tuple[float, float, float]]:
    n = len(pos)
    if len(masses) != n:
        raise ValueError("masses length mismatch")
    acc = [[0.0, 0.0, 0.0] for _ in range(n)]

    for i in range(n):
        xi, yi, zi = pos[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj, zj = pos[j]
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            r2 = dx * dx + dy * dy + dz * dz + eps
            inv_r3 = 1.0 / (r2 * (r2**0.5))
            s = g * masses[j] * inv_r3
            acc[i][0] += s * dx
            acc[i][1] += s * dy
            acc[i][2] += s * dz
    return [(a[0], a[1], a[2]) for a in acc]
