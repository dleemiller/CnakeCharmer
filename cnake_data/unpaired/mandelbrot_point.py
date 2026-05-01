"""Mandelbrot set membership test for a single point."""

from __future__ import annotations


def mandel(real, imag, max_iterations=20):
    z_real = 0.0
    z_imag = 0.0
    for _ in range(max_iterations):
        z_real, z_imag = (
            z_real * z_real - z_imag * z_imag + real,
            2.0 * z_real * z_imag + imag,
        )
        if (z_real * z_real + z_imag * z_imag) >= 4.0:
            return False
    return (z_real * z_real + z_imag * z_imag) < 4.0
