"""Masked PSNR between two uint8 images."""

from __future__ import annotations

import math


def psnr_masked(img1, img2):
    h = len(img1)
    w = len(img1[0])
    n_pixel = w * h
    total = 0.0

    for i in range(h):
        for j in range(w):
            if img1[i][j] == 0 or img2[i][j] == 0:
                n_pixel -= 1
                continue
            diff = abs(int(img1[i][j]) - int(img2[i][j]))
            total += float(diff * diff)

    if n_pixel <= 0:
        return float("inf")
    error = total / n_pixel
    if error == 0.0:
        return float("inf")
    return 10 * math.log10((255**2) / error)
