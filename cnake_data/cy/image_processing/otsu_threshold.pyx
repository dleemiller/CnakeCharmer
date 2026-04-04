# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Otsu's method for automatic image thresholding (Cython-optimized).

Computes the optimal threshold that minimizes intra-class variance
for a deterministic grayscale image.

Keywords: image processing, Otsu, threshold, segmentation, histogram, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def otsu_threshold(int n):
    """Compute Otsu threshold for an n x n deterministic grayscale image."""
    cdef int total_pixels = n * n
    cdef int i, j, t, val
    cdef int best_threshold = 0
    cdef double best_variance = 0.0
    cdef int w0 = 0, w1
    cdef double sum0 = 0.0, sum1, total_sum
    cdef double mean0, mean1, variance
    cdef int foreground_count

    # Build histogram
    cdef int *hist = <int *>malloc(256 * sizeof(int))
    if not hist:
        raise MemoryError()

    for i in range(256):
        hist[i] = 0

    for i in range(n):
        for j in range(n):
            val = (i * 7 + j * 13 + i * j * 3 + 5) % 256
            hist[val] += 1

    # Compute total mean
    total_sum = 0.0
    for t in range(256):
        total_sum += t * hist[t]

    w0 = 0
    sum0 = 0.0

    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total_pixels - w0
        if w1 == 0:
            break

        sum0 += t * hist[t]
        sum1 = total_sum - sum0

        mean0 = sum0 / w0
        mean1 = sum1 / w1

        variance = <double>w0 * <double>w1 * (mean0 - mean1) * (mean0 - mean1)

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    # Count foreground pixels (above threshold)
    foreground_count = 0
    for t in range(best_threshold + 1, 256):
        foreground_count += hist[t]

    free(hist)
    return (best_threshold, foreground_count, best_variance)
