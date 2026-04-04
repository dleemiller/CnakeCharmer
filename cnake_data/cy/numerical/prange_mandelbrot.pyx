# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel Mandelbrot set computation using prange.

Keywords: numerical, mandelbrot, fractal, prange, parallel, cython, benchmark
"""

from cython.parallel import prange
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def prange_mandelbrot(int n):
    """Count Mandelbrot set points on n x n grid with prange."""
    cdef int row, col, k, in_set
    cdef int max_iter = 100
    cdef double cr, ci, zr, zi, zr2, zi2
    cdef int count = 0

    for row in prange(n, nogil=True):
        ci = -1.5 + 3.0 * row / n
        for col in range(n):
            cr = -2.0 + 3.0 * col / n
            zr = 0.0
            zi = 0.0
            in_set = 1
            for k in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    in_set = 0
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
            count += in_set

    return count
