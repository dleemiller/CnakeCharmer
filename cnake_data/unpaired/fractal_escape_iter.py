from __future__ import annotations


def mandelbrot_escape(cr: float, ci: float, max_iter: int) -> int:
    zr = 0.0
    zi = 0.0
    for i in range(max_iter):
        zr2 = zr * zr - zi * zi + cr
        zi2 = 2.0 * zr * zi + ci
        zr, zi = zr2, zi2
        if zr * zr + zi * zi > 4.0:
            return i + 1
    return max_iter


def mandelbrot_grid(
    x0: float, x1: float, y0: float, y1: float, w: int, h: int, max_iter: int
) -> list[list[int]]:
    out = [[0] * w for _ in range(h)]
    for iy in range(h):
        ci = y0 + (y1 - y0) * iy / max(1, h - 1)
        for ix in range(w):
            cr = x0 + (x1 - x0) * ix / max(1, w - 1)
            out[iy][ix] = mandelbrot_escape(cr, ci, max_iter)
    return out
