"""Mandelbrot escape-iteration grid generation."""

from __future__ import annotations


def _mandelbrot(creal: float, cimag: float, maxiter: int) -> int:
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real * real
        imag2 = imag * imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2.0 * real * imag + cimag
        real = real2 - imag2 + creal
    return maxiter


def mandelbrot_set(
    xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, maxiter: int
) -> list[list[int]]:
    out = [[0 for _ in range(width)] for _ in range(height)]
    for j in range(height):
        cimag = ymin + (ymax - ymin) * (j / max(1, height - 1))
        for i in range(width):
            creal = xmin + (xmax - xmin) * (i / max(1, width - 1))
            out[j][i] = _mandelbrot(creal, cimag, maxiter)
    return out
