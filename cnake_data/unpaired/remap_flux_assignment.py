"""Grid remap helpers for assignment and flux-sifting style updates."""

from __future__ import annotations


def assignment(
    f_old: list[list[float]],
    map1: list[list[int]],
    map2: list[list[int]],
    dim1: int,
    dim2: int,
) -> list[list[float]]:
    f_new = [[0.0 for _ in range(dim2)] for _ in range(dim1)]
    for i in range(dim1):
        for j in range(dim2):
            f_new[i][j] = f_old[map1[i][j]][map2[i][j]]
    return f_new


def sift_flux(flux: list[list[float]], idx_map: list[list[int]]) -> list[list[float]]:
    d1 = len(flux)
    d2 = len(flux[0]) if d1 else 0
    out = [[0.0 for _ in range(d2)] for _ in range(d1)]
    for i in range(d1):
        for j in range(d2):
            jj = idx_map[i][j]
            if 0 <= jj < d2:
                out[i][jj] += flux[i][j]
    return out


def sift_old(f_old: list[list[float]], keep_mask: list[list[int]]) -> list[list[float]]:
    d1 = len(f_old)
    d2 = len(f_old[0]) if d1 else 0
    out = [[0.0 for _ in range(d2)] for _ in range(d1)]
    for i in range(d1):
        for j in range(d2):
            if keep_mask[i][j]:
                out[i][j] = f_old[i][j]
    return out
