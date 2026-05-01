from __future__ import annotations

import cmath
import math


def bitrev_shuffle(x: list[complex]) -> None:
    n = len(x)
    j = 0
    for i in range(1, n):
        b = n >> 1
        while j >= b:
            j -= b
            b >>= 1
        j += b
        if j > i:
            x[i], x[j] = x[j], x[i]


def fft_in_place(x: list[complex]) -> None:
    n = len(x)
    bitrev_shuffle(x)
    trans = 2
    while trans <= n:
        wb_step = cmath.exp(-2j * math.pi / trans)
        wb = 1 + 0j
        half = trans >> 1
        for t in range(half):
            for tr in range(n // trans):
                i = tr * trans + t
                j = i + half
                a = x[i]
                b = x[j] * wb
                x[i] = a + b
                x[j] = a - b
            wb *= wb_step
        trans <<= 1


def fft(x: list[complex]) -> list[complex]:
    y = x[:]
    fft_in_place(y)
    return y
