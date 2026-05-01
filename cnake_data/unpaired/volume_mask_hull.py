from __future__ import annotations


def point_in_hull(
    i: int,
    j: int,
    k: int,
    hull_eqs: list[tuple[float, float, float, float]],
    tolerance: float = 1e-6,
) -> bool:
    return all(i * a + j * b + k * c + d <= tolerance for a, b, c, d in hull_eqs)


def mask_array(
    shape: tuple[int, int, int], hull_eqs: list[tuple[float, float, float, float]], val: int = 1
) -> tuple[list[list[list[int]]], int]:
    ni, nj, nk = shape
    out = [[[0 for _ in range(nk)] for _ in range(nj)] for _ in range(ni)]
    cnt = 0
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if point_in_hull(i, j, k, hull_eqs):
                    out[i][j][k] = val
                    cnt += 1
    return out, cnt
