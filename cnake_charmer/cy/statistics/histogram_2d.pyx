# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D histogram of deterministic point pairs (Cython-optimized).

Keywords: statistics, histogram, 2d, binning, cython, benchmark
"""

from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def histogram_2d(int n):
    """Compute 2D histogram and return the maximum bin count using C array."""
    cdef int bins[100]  # 10x10 flattened
    cdef int i, xval, yval, bx, by, max_count

    memset(bins, 0, 100 * sizeof(int))

    for i in range(n):
        xval = (i * 7 + 3) % 100
        yval = (i * 13 + 7) % 100
        bx = xval / 10
        by = yval / 10
        bins[bx * 10 + by] += 1

    max_count = 0
    for i in range(100):
        if bins[i] > max_count:
            max_count = bins[i]

    return max_count
