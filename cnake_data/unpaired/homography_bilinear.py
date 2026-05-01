from __future__ import annotations

import math


def get_pixel(image: list[list[float]], r: int, c: int, cval: float = 0.0) -> float:
    rows = len(image)
    cols = len(image[0]) if rows else 0
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return cval
    return image[r][c]


def bilinear(image: list[list[float]], r: float, c: float, cval: float = 0.0) -> float:
    minr = math.floor(r)
    minc = math.floor(c)
    maxr = math.ceil(r)
    maxc = math.ceil(c)
    dr = r - minr
    dc = c - minc
    top = (1.0 - dc) * get_pixel(image, minr, minc, cval) + dc * get_pixel(image, minr, maxc, cval)
    bottom = (1.0 - dc) * get_pixel(image, maxr, minc, cval) + dc * get_pixel(
        image, maxr, maxc, cval
    )
    return (1.0 - dr) * top + dr * bottom


def apply_homography(image: list[list[float]], h_inv: list[list[float]]) -> list[list[float]]:
    out_r = len(image)
    out_c = len(image[0]) if out_r else 0
    out = [[0.0 for _ in range(out_c)] for _ in range(out_r)]
    for tr in range(out_r):
        for tc in range(out_c):
            xx = h_inv[0][0] * tc + h_inv[0][1] * tr + h_inv[0][2]
            yy = h_inv[1][0] * tc + h_inv[1][1] * tr + h_inv[1][2]
            zz = h_inv[2][0] * tc + h_inv[2][1] * tr + h_inv[2][2]
            c = xx / zz
            r = yy / zz
            out[tr][tc] = bilinear(image, r, c, 0.0)
    return out
