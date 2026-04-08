# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Radial vignetting model for optical systems — Cython implementation."""

from libc.math cimport sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200, 200))
def vignetting_model(int rows, int cols):
    """Apply cos^4 + polynomial vignetting to a pixel grid."""
    cdef double cx = (cols - 1) * 0.5
    cdef double cy = (rows - 1) * 0.5
    cdef double half_diag = sqrt(cx * cx + cy * cy)
    cdef double a0 = 1.0, a2 = -0.35, a4 = 0.08
    cdef double total = 0.0, min_gain = 2.0, center_gain = 0.0
    cdef double dy, dx, r2, cos_theta, cos4, poly, gain
    cdef int r, c

    for r in range(rows):
        dy = (r - cy) / half_diag
        for c in range(cols):
            dx = (c - cx) / half_diag
            r2 = dx * dx + dy * dy
            cos_theta = 1.0 / sqrt(1.0 + r2)
            cos4 = cos_theta * cos_theta * cos_theta * cos_theta
            poly = a0 + a2 * r2 + a4 * r2 * r2
            gain = cos4 * poly
            total += gain
            if gain < min_gain:
                min_gain = gain
            if r == rows // 2 and c == cols // 2:
                center_gain = gain

    return (total, min_gain, center_gain)
