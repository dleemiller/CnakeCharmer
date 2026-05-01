from __future__ import annotations


def particle_center_of_mass(
    pos: list[tuple[float, float, float]], mass: list[float]
) -> tuple[float, float, float]:
    if len(pos) != len(mass):
        raise ValueError("length mismatch")
    sx = sy = sz = sm = 0.0
    for (x, y, z), m in zip(pos, mass, strict=False):
        sx += x * m
        sy += y * m
        sz += z * m
        sm += m
    if sm == 0.0:
        return 0.0, 0.0, 0.0
    return sx / sm, sy / sm, sz / sm


def radial_distances(
    pos: list[tuple[float, float, float]], center: tuple[float, float, float]
) -> list[float]:
    cx, cy, cz = center
    out: list[float] = []
    for x, y, z in pos:
        dx = x - cx
        dy = y - cy
        dz = z - cz
        out.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    return out
