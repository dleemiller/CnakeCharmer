from __future__ import annotations


def triangle_area(
    p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5


def mesh_area(vertices: list[tuple[float, float]], tris: list[tuple[int, int, int]]) -> float:
    s = 0.0
    for i, j, k in tris:
        s += triangle_area(vertices[i], vertices[j], vertices[k])
    return s
