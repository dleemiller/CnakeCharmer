"""Find bounding crop area of non-zero pixels."""

from __future__ import annotations

import numpy as np


def find_white(img: np.ndarray, i: int, j: int, mode: int):
    if mode:
        for k in range(0, j):
            if img[i][k] > 0:
                return True
    else:
        for k in range(0, j):
            if img[k][i] > 0:
                return True
    return False


def find_min_point(
    img: np.ndarray, min_v: int, max_v: int, inner_len: int, mode: int, reverse: int
):
    if reverse:
        for i in range(min_v - 1, max_v - 1, -1):
            if find_white(img, i, inner_len, mode):
                return i
    else:
        for i in range(min_v, max_v):
            if find_white(img, i, inner_len, mode):
                return i
    return -1


def get_crop_area(img: np.ndarray):
    rows = img.shape[0]
    columns = img.shape[1]
    y_min = find_min_point(img, 0, rows, columns, 1, 0)
    y_max = find_min_point(img, rows, 0, columns, 1, 1)
    x_min = find_min_point(img, 0, columns, rows, 0, 0)
    x_max = find_min_point(img, columns, 0, rows, 0, 1)
    return y_min, y_max, x_min, x_max
