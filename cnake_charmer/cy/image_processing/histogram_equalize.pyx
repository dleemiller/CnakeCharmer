# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute histogram equalization on an n x n grayscale image (Cython-optimized).

Keywords: image processing, histogram equalization, contrast, CDF, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def histogram_equalize(int n):
    """Perform histogram equalization and return sum of equalized values."""
    cdef int i, j, size, equalized
    cdef long long total
    cdef int cdf_min, denom

    size = n * n

    cdef int *img = <int *>malloc(size * sizeof(int))
    cdef int *hist = <int *>malloc(256 * sizeof(int))
    cdef int *cdf = <int *>malloc(256 * sizeof(int))
    if not img or not hist or not cdf:
        free(img)
        free(hist)
        free(cdf)
        raise MemoryError()

    # Generate image
    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 17 + j * 31 + 5) % 256

    # Compute histogram
    memset(hist, 0, 256 * sizeof(int))
    for i in range(size):
        hist[img[i]] += 1

    # Compute CDF
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Find min non-zero CDF
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    # Equalize and sum
    total = 0
    denom = size - cdf_min
    if denom == 0:
        denom = 1
    for i in range(size):
        equalized = ((cdf[img[i]] - cdf_min) * 255 + denom // 2) // denom
        total += equalized

    free(img)
    free(hist)
    free(cdf)
    return int(total)
