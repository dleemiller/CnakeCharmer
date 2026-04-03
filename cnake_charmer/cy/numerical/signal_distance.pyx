# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Euclidean distance between signal arrays (Cython-optimized).

Keywords: euclidean, distance, signal, sqrt, libc, numerical, pairwise, cython
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

N = 5000


@cython_benchmark(syntax="cy", args=(N,))
def signal_distance(int n):
    """Compute pairwise Euclidean distances for n signal pairs of length 64."""
    cdef int sig_len = 64
    cdef int total_signals = 2 * n
    cdef int i, j
    cdef double total_distance = 0.0
    cdef double max_distance = 0.0
    cdef double sq_sum, diff, dist
    cdef double *signals
    cdef double *sig_a
    cdef double *sig_b

    signals = <double *>malloc(total_signals * sig_len * sizeof(double))
    if signals == NULL:
        raise MemoryError("Failed to allocate signal arrays")

    # Generate signals deterministically
    for i in range(total_signals):
        for j in range(sig_len):
            signals[i * sig_len + j] = ((i * 7 + j * 13 + 42) % 1000) / 100.0

    with nogil:
        for i in range(n):
            sig_a = &signals[(2 * i) * sig_len]
            sig_b = &signals[(2 * i + 1) * sig_len]

            sq_sum = 0.0
            for j in range(sig_len):
                diff = sig_a[j] - sig_b[j]
                sq_sum += diff * diff

            dist = sqrt(sq_sum)

            total_distance += dist
            if dist > max_distance:
                max_distance = dist

    free(signals)

    return (total_distance, max_distance)
