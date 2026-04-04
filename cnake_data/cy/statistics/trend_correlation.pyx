# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Trend-preserving correlation of a deterministic matrix (Cython-optimized).

Uses malloc/free for trend arrays and nogil for the hot loops.

Keywords: statistics, trend, correlation, hamming, pairwise, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef double hamming_correlation(double *x, double *y, unsigned int length) nogil:
    cdef double same = 0.0
    cdef double diff = 0.0
    cdef unsigned int k
    for k in range(length):
        if x[k] == y[k]:
            same += 1.0
        else:
            diff += 1.0
    cdef double ratio_same = same / length
    cdef double ratio_diff = diff / length
    if ratio_same > ratio_diff:
        return ratio_same
    return ratio_diff


@cython_benchmark(syntax="cy", args=(200,))
def trend_correlation(int n):
    """Compute trend-preserving correlation of an n x 15 deterministic matrix.

    Args:
        n: Number of rows in the matrix.

    Returns:
        Tuple of (average_correlation, trend_sum).
    """
    cdef unsigned int cols = 15
    cdef unsigned int trend_len = cols - 1
    cdef unsigned int i, j
    cdef double val_cur, val_nxt
    cdef double trend_sum = 0.0
    cdef double total_corr = 0.0
    cdef unsigned int pair_count = 0
    cdef double avg_corr
    cdef double **trends = <double **>malloc(n * sizeof(double *))

    if trends == NULL:
        return (0.0, 0.0)

    # Build trend vectors
    for i in range(n):
        trends[i] = <double *>malloc(trend_len * sizeof(double))
        for j in range(trend_len):
            val_cur = ((i * 17 + j * 31 + 5) % 1000) / 100.0
            val_nxt = ((i * 17 + (j + 1) * 31 + 5) % 1000) / 100.0
            if val_nxt - val_cur > 0:
                trends[i][j] = 1.0
            else:
                trends[i][j] = -1.0
            trend_sum += trends[i][j]

    # Pairwise Hamming correlation
    with nogil:
        for i in range(n - 1):
            for j in range(i + 1, n):
                total_corr += hamming_correlation(trends[i], trends[j], trend_len)
                pair_count += 1

    # Cleanup
    for i in range(n):
        free(trends[i])
    free(trends)

    avg_corr = total_corr / pair_count
    return (avg_corr, trend_sum)
