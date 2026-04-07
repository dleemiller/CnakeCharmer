# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigma filter on a 2D float image — Cython implementation."""

from libc.math cimport fabs, sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(60, 80, 3, 0.5))
def sigma_filter_2d(int n, int m, int radius, double threshold):
    """Apply sigma filter to a deterministically generated n×m image.

    Args:
        n: Image height (rows).
        m: Image width (columns).
        radius: Half-window size.
        threshold: Pixel inclusion threshold.

    Returns:
        Tuple of (total_sum, top_left_pixel, center_pixel).
    """
    cdef int y, x, dj, di, yy, xx, count
    cdef double center, val, acc
    cdef double *src = <double *>malloc(n * m * sizeof(double))
    cdef double *dst = <double *>malloc(n * m * sizeof(double))
    if not src or not dst:
        free(src); free(dst)
        raise MemoryError()

    # Generate deterministic source image
    for y in range(n):
        for x in range(m):
            src[y * m + x] = sin(y * 0.5 + x * 0.7)

    # Apply sigma filter
    for y in range(n):
        for x in range(m):
            center = src[y * m + x]
            acc = 0.0
            count = 0
            for dj in range(-radius, radius + 1):
                yy = y + dj
                if yy < 0:
                    yy = 0
                elif yy >= n:
                    yy = n - 1
                for di in range(-radius, radius + 1):
                    xx = x + di
                    if xx < 0:
                        xx = 0
                    elif xx >= m:
                        xx = m - 1
                    val = src[yy * m + xx]
                    if fabs(center - val) < threshold:
                        acc += val
                        count += 1
            dst[y * m + x] = acc / count if count > 0 else 0.0

    cdef double top_left = dst[0]
    cdef double center_px = dst[(n // 2) * m + m // 2]
    cdef double quarter_px = dst[(n // 4) * m + m // 4]
    free(src)
    free(dst)
    return (top_left, center_px, quarter_px)
