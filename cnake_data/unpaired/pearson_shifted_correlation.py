"""Pearson correlation between two images with integer shifts."""

from __future__ import annotations


def _pearson_correlation(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = 0.0
    va = 0.0
    vb = 0.0
    for i in range(n):
        da = a[i] - ma
        db = b[i] - mb
        num += da * db
        va += da * da
        vb += db * db
    den = (va * vb) ** 0.5
    return 0.0 if den == 0.0 else num / den


def calculate_ppmcc(im1, im2, shift_x, shift_y):
    h = len(im1)
    w = len(im1[0]) if h else 0
    new_w = int(w - abs(shift_x))
    new_h = int(h - abs(shift_y))
    x0 = max(0, -shift_x)
    y0 = max(0, -shift_y)
    x1 = x0 + shift_x
    y1 = y0 + shift_y

    a = []
    b = []
    for y in range(new_h):
        for x in range(new_w):
            a.append(im1[y0 + y][x0 + x])
            b.append(im2[y1 + y][x1 + x])
    return _pearson_correlation(a, b)
