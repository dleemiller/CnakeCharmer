# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Brown-Conrady radial and tangential lens distortion model — Cython implementation."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200, 200))
def radial_distortion(int rows, int cols):
    """Apply Brown-Conrady distortion to a normalized coordinate grid."""
    cdef double k1 = -0.28, k2 = 0.08, p1 = 0.001, p2 = -0.0015
    cdef double sum_xd = 0.0, sum_yd = 0.0, max_r2 = 0.0
    cdef double x, y, r2, radial, xd, yd
    cdef int r, c

    for r in range(rows):
        y = -1.0 + 2.0 * r / (rows - 1)
        for c in range(cols):
            x = -1.0 + 2.0 * c / (cols - 1)
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            sum_xd += xd
            sum_yd += yd
            if r2 > max_r2:
                max_r2 = r2

    return (sum_xd, sum_yd, max_r2)
