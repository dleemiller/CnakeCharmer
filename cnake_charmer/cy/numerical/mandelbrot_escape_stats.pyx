# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Mandelbrot membership/escape statistics (Cython).

Sourced from SFT DuckDB blob: 6b0c1ba7b17cea70784b6fa6f3d31831b357ba46
Keywords: mandelbrot, fractal, escape time, numerical, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(350, 220, 64))
def mandelbrot_escape_stats(int width, int height, int max_iter):
    cdef int ix, iy, it
    cdef int inside = 0
    cdef int edge_inside = 0
    cdef int escape_sum = 0
    cdef int escaped_at
    cdef double min_x = -2.0
    cdef double min_y = -1.2
    cdef double dx = 3.0 / width
    cdef double dy = 2.4 / height
    cdef double real, imag, zr, zi, zrn

    for ix in range(width):
        real = min_x + ix * dx
        for iy in range(height):
            imag = min_y + iy * dy
            zr = 0.0
            zi = 0.0
            escaped_at = 0
            for it in range(1, max_iter + 1):
                zrn = zr * zr - zi * zi + real
                zi = 2.0 * zr * zi + imag
                zr = zrn
                if zr * zr + zi * zi >= 4.0:
                    escaped_at = it
                    break

            if escaped_at == 0:
                inside += 1
                if ix == 0 or iy == 0 or ix == width - 1 or iy == height - 1:
                    edge_inside += 1
            else:
                escape_sum += escaped_at

    return (inside, escape_sum, edge_inside)

