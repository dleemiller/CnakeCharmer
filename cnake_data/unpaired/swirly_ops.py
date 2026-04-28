"""Image-mask chunking and rubix-like shift operations."""

from __future__ import annotations

import numpy as np


def chunk_select(indices: np.ndarray):
    contiguous = np.diff(indices) == 1
    shortest = contiguous.size
    longest = indices.size
    out = []
    i = -1
    while i < longest - 1:
        i += 1
        left = int(indices[i])
        if i >= shortest or not contiguous[i]:
            out.append((left, left + 1))
        else:
            while i < longest - 1:
                i += 1
                right = int(indices[i])
                if i >= shortest or not contiguous[i]:
                    out.append((left, right + 1))
                    break
    return out


def column_avgs(select: np.ndarray):
    nrows, ncols = select.shape
    avgs = np.empty((nrows,), dtype=np.float64)
    avgs[:] = -1.0
    total_sum = 0.0
    total_count = 0
    for i in range(nrows):
        current_sum = 0.0
        current_count = 0
        for j in range(ncols):
            value = int(select[i, j])
            current_sum += j * value
            current_count += value
        total_sum += current_sum
        total_count += current_count
        if current_count:
            avgs[i] = current_sum / current_count
    total_avg = total_sum / total_count if total_count else -1.0
    return avgs, total_avg


def move_rubix3d(img: np.ndarray, travel: int):
    cols, rows, chans = img.shape
    tail = img[cols - travel : cols].copy()
    img[travel:cols] = img[: cols - travel]
    img[:travel] = tail


def move_rubix2d(img: np.ndarray, travel: int):
    cols, rows = img.shape
    tail = img[cols - travel : cols].copy()
    img[travel:cols] = img[: cols - travel]
    img[:travel] = tail


def move_rubix(img: np.ndarray, select: np.ndarray, travel: int):
    if travel < 0:
        img = img[::-1]
        select = select[::-1]
        travel = -travel
    move_rubix3d(img, travel)
    move_rubix2d(select, travel)
    return img, select
