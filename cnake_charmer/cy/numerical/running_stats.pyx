# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sliding window statistics (min, max, mean, variance) over a sequence (Cython-optimized).

Keywords: numerical, sliding window, statistics, running stats, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def running_stats(int n):
    """Compute sliding window stats (w=100) over a deterministic sequence.

    Sequence: v[i] = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0
    Returns stats at the last window position i=n-1:
        (int(mean * 1e9), int(variance * 1e9), int(min * 1e9), int(max * 1e9))
    """
    cdef int w = 100
    cdef double *v = <double *>malloc(n * sizeof(double))
    if not v:
        raise MemoryError()

    cdef int i, j
    cdef unsigned int lcg
    cdef double window_sum, window_sum_sq
    cdef double mean, variance, win_min, win_max
    cdef double last_mean, last_variance, last_min, last_max
    cdef double val_new, val_old

    with nogil:
        # Generate the sequence
        for i in range(n):
            lcg = (<unsigned int>(i * 1664525) + <unsigned int>1013904223)
            v[i] = lcg / 4294967296.0

        # Prime the first window
        window_sum = 0.0
        window_sum_sq = 0.0
        for i in range(w):
            window_sum += v[i]
            window_sum_sq += v[i] * v[i]

        last_mean = 0.0
        last_variance = 0.0
        last_min = 0.0
        last_max = 0.0

        for i in range(w - 1, n):
            if i > w - 1:
                val_new = v[i]
                val_old = v[i - w]
                window_sum += val_new - val_old
                window_sum_sq += val_new * val_new - val_old * val_old

            mean = window_sum / w
            variance = window_sum_sq / w - mean * mean

            # Scan window for min/max
            win_min = v[i - w + 1]
            win_max = v[i - w + 1]
            for j in range(i - w + 2, i + 1):
                if v[j] < win_min:
                    win_min = v[j]
                if v[j] > win_max:
                    win_max = v[j]

            last_mean = mean
            last_variance = variance
            last_min = win_min
            last_max = win_max

    free(v)
    return (
        int(last_mean * 1e9),
        int(last_variance * 1e9),
        int(last_min * 1e9),
        int(last_max * 1e9),
    )
