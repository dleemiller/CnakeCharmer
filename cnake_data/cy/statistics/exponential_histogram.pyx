# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute entropy of histogram with exponential bin widths (Cython-optimized).

Keywords: histogram, entropy, exponential bins, statistics, information theory, cython, benchmark
"""

from libc.math cimport log
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def exponential_histogram(int n):
    """Compute entropy of an exponential-width histogram."""
    cdef int max_val = 10000
    # Bins: [0,1), [1,2), [2,4), [4,8), ..., up to covering max_val
    # Number of bins: 1 (for [0,1)) + ceil(log2(max_val)) = ~14
    cdef int bin_edges[20]
    cdef int num_bins, edge, i
    cdef int v, lo, hi, mid

    bin_edges[0] = 0
    bin_edges[1] = 1
    num_bins = 1
    edge = 1
    while edge < max_val:
        edge *= 2
        num_bins += 1
        bin_edges[num_bins] = edge

    cdef int *counts = <int *>malloc(num_bins * sizeof(int))
    if not counts:
        raise MemoryError()
    memset(counts, 0, num_bins * sizeof(int))

    for i in range(n):
        v = (i * 17 + 5) % max_val
        # Binary search for bin
        lo = 0
        hi = num_bins - 1
        while lo < hi:
            mid = (lo + hi) / 2
            if v < bin_edges[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        counts[lo] += 1

    # Compute entropy
    cdef double entropy = 0.0
    cdef double p
    cdef double dn = <double>n
    for i in range(num_bins):
        if counts[i] > 0:
            p = <double>counts[i] / dn
            entropy -= p * log(p)

    free(counts)

    return entropy
