# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Projective homography warp on a pixel grid — Cython implementation."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(120, 120))
def homography_warp(int rows, int cols):
    """Apply a fixed 3x3 homography to a pixel grid and return warp statistics."""
    cdef double h00 = 1.05, h01 = 0.02, h02 = 5.0
    cdef double h10 = -0.01, h11 = 0.98, h12 = 3.0
    cdef double h20 = 0.0001, h21 = 0.00005, h22 = 1.0

    cdef double sum_dx = 0.0
    cdef double sum_dy = 0.0
    cdef int count = 0
    cdef double w, dx, dy
    cdef int r, c

    for r in range(rows):
        for c in range(cols):
            w = h20 * c + h21 * r + h22
            dx = (h00 * c + h01 * r + h02) / w
            dy = (h10 * c + h11 * r + h12) / w
            if 0.0 <= dx < cols and 0.0 <= dy < rows:
                sum_dx += dx
                sum_dy += dy
                count += 1

    return (sum_dx, sum_dy, count)
