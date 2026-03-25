# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Mandelbrot set point counting (Cython-optimized).

Keywords: numerical, mandelbrot, fractal, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def mandelbrot_count(int n):
    """Count Mandelbrot set points in an n x n grid using pure C arithmetic."""
    cdef int row, col, iteration, count
    cdef int max_iter = 100
    cdef double cr, ci, zr, zi, zr2, zi2
    cdef double x_min = -2.0
    cdef double x_range = 3.0
    cdef double y_min = -1.5
    cdef double y_range = 3.0
    cdef double scale = 1.0 / (n - 1)

    count = 0

    for row in range(n):
        ci = y_min + row * y_range * scale
        for col in range(n):
            cr = x_min + col * x_range * scale
            zr = 0.0
            zi = 0.0
            iteration = 0
            while iteration < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                iteration += 1
            if iteration == max_iter:
                count += 1

    return count
