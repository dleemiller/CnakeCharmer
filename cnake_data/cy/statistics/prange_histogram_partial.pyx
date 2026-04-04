# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel partial histograms using prange over chunks.

Keywords: statistics, histogram, prange, parallel, chunked, cython, benchmark
"""

from cython.parallel import prange
from libc.stdlib cimport calloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def prange_histogram_partial(int n):
    """Build histogram with prange over chunks, return max."""
    cdef int num_bins = 256
    cdef int num_chunks = 64
    cdef int chunk_size = (n + num_chunks - 1) // num_chunks
    cdef int i, c, b, start, end, val
    cdef int max_count = 0

    # Allocate partial histograms: num_chunks x num_bins
    cdef int *partials = <int *>calloc(
        num_chunks * num_bins, sizeof(int)
    )
    cdef int *final_bins = <int *>calloc(
        num_bins, sizeof(int)
    )
    if not partials or not final_bins:
        free(partials)
        free(final_bins)
        raise MemoryError()

    for c in prange(num_chunks, nogil=True):
        start = c * chunk_size
        end = start + chunk_size
        if end > n:
            end = n
        for i in range(start, end):
            val = (
                (
                    <int>(
                        (
                            <long long>i
                            * <long long>2654435761
                            + 17
                        ) >> 8
                    )
                ) & 255
            )
            partials[c * num_bins + val] += 1

    # Merge partial histograms
    for b in range(num_bins):
        for c in range(num_chunks):
            final_bins[b] += partials[c * num_bins + b]

    for b in range(num_bins):
        if final_bins[b] > max_count:
            max_count = final_bins[b]

    free(partials)
    free(final_bins)
    return max_count
