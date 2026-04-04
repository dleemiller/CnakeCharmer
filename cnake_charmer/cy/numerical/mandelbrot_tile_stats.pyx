# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute escape statistics over a Mandelbrot tile (Cython)."""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(-2.0, 1.0, 380, 120, 4.0))
def mandelbrot_tile_stats(double min_val, double max_val, int size, int max_iter, double threshold):
    cdef double step = (abs(min_val) + max_val) / (size - 1)
    cdef int i, j, it, escaped
    cdef double imag, real, zr, zi, zr2, zi2
    cdef double sum_escape = 0.0
    cdef int stable_count = 0
    cdef double edge_checksum = 0.0

    for i in range(size):
        imag = max_val - step * i
        for j in range(size):
            real = min_val + step * j
            zr = 0.0
            zi = 0.0
            escaped = max_iter
            for it in range(max_iter):
                zr2 = zr * zr - zi * zi + real
                zi2 = 2.0 * zr * zi + imag
                zr = zr2
                zi = zi2
                if zr * zr + zi * zi > threshold:
                    escaped = it + 1
                    break
            sum_escape += escaped
            if escaped == max_iter:
                stable_count += 1
            if i == 0 or j == 0 or i == size - 1 or j == size - 1:
                edge_checksum += escaped * (1.0 + ((i + j) & 1) * 0.25)

    return (sum_escape, <double>stable_count, edge_checksum)
