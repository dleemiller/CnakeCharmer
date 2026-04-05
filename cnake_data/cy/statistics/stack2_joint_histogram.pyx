# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build a joint histogram for deterministic paired signals (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: b67354d664bd9d79078d1fc8f8bb0e617479e227
- filename: histogram_2D.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(260000, 48, 23))
def stack2_joint_histogram(int sample_count, int bin_count, int scale_tag):
    cdef unsigned int state = <unsigned int>((1357911 + scale_tag * 4099) & 0xFFFFFFFF)
    cdef int hist_size = bin_count * bin_count
    cdef int *hist = <int *>malloc(hist_size * sizeof(int))
    cdef int idx, row, col, bx, by, val
    cdef unsigned int left, right
    cdef int max_bin = 0
    cdef int diag_sum = 0
    cdef unsigned int checksum = 0

    if not hist:
        raise MemoryError()

    for idx in range(hist_size):
        hist[idx] = 0

    for idx in range(sample_count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        left = (state >> 4) & 0xFFFF
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        right = (state >> 7) & 0xFFFF
        bx = <int>((left * <unsigned int>bin_count) >> 16)
        by = <int>((right * <unsigned int>bin_count) >> 16)
        hist[bx * bin_count + by] += 1

    for row in range(bin_count):
        for col in range(bin_count):
            val = hist[row * bin_count + col]
            if val > max_bin:
                max_bin = val
            if row == col:
                diag_sum += val
            checksum = (checksum + <unsigned int>(val * (row + 1) * (col + 3))) & 0xFFFFFFFF

    val = hist[bin_count - 1]
    free(hist)
    return (max_bin, diag_sum, checksum, val)
