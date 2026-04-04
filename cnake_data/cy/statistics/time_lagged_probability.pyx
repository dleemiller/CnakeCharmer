# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Time-lagged conditional probability estimation (Cython-optimized).

Keywords: probability, conditional, time series, lag, statistics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark
from libc.stdlib cimport malloc, free


@cython_benchmark(syntax="cy", args=(500000,))
def time_lagged_probability(int n):
    """Estimate time-lagged conditional probabilities over n timesteps."""
    cdef int delta = 3
    cdef int t
    cdef int count_a = 0, count_b = 0
    cdef int count_ab = 0, count_ba = 0
    cdef double p_ab, p_ba

    cdef int *event_a = <int *>malloc(n * sizeof(int))
    cdef int *event_b = <int *>malloc(n * sizeof(int))

    if event_a == NULL or event_b == NULL:
        if event_a != NULL:
            free(event_a)
        if event_b != NULL:
            free(event_b)
        return (0.0, 0.0)

    # Compute event arrays
    for t in range(n):
        if (t * 7 + 3) % 5 == 0:
            event_a[t] = 1
        else:
            event_a[t] = 0
        if (t * 11 + 7) % 7 == 0:
            event_b[t] = 1
        else:
            event_b[t] = 0

    # Count marginals
    for t in range(n):
        count_a += event_a[t]
        count_b += event_b[t]

    # Count joint with lag
    for t in range(delta, n):
        if event_b[t - delta] == 1 and event_a[t] == 1:
            count_ab += 1
        if event_a[t - delta] == 1 and event_b[t] == 1:
            count_ba += 1

    free(event_a)
    free(event_b)

    if count_b > 0:
        p_ab = <double>count_ab / <double>count_b
    else:
        p_ab = 0.0
    if count_a > 0:
        p_ba = <double>count_ba / <double>count_a
    else:
        p_ba = 0.0

    return (p_ab, p_ba)
