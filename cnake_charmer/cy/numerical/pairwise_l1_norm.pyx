# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pairwise L1 distance matrix computation (Cython-optimized).

Uses flat C arrays and nogil for maximum performance on the triple
nested loop computing Manhattan distances between point sets.

Keywords: pairwise, distance, L1, manhattan, numerical, benchmark, cython
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.stdlib cimport malloc, free
from libc.math cimport fabs


@cython_benchmark(syntax="cy", args=(80, 80, 10))
def pairwise_l1_norm(int n1, int n2, int k):
    """Compute pairwise L1 distance matrix between two point sets."""
    cdef int i, j, d
    cdef double dist, total_sum, max_distance, distance_at_mid_mid
    cdef int mid_i, mid_j
    cdef double *d1 = <double *>malloc(n1 * k * sizeof(double))
    cdef double *d2 = <double *>malloc(n2 * k * sizeof(double))
    cdef double *result = <double *>malloc(n1 * n2 * sizeof(double))

    if d1 == NULL or d2 == NULL or result == NULL:
        if d1 != NULL:
            free(d1)
        if d2 != NULL:
            free(d2)
        if result != NULL:
            free(result)
        raise MemoryError("Failed to allocate arrays")

    # Generate deterministic point data from indices
    for i in range(n1):
        for d in range(k):
            d1[i * k + d] = ((i * 31 + d * 7 + 13) % 1000) / 100.0

    for j in range(n2):
        for d in range(k):
            d2[j * k + d] = ((j * 37 + d * 11 + 17) % 1000) / 100.0

    with nogil:
        # Compute pairwise L1 distance matrix
        for i in range(n1):
            for j in range(n2):
                dist = 0.0
                for d in range(k):
                    dist += fabs(d1[i * k + d] - d2[j * k + d])
                result[i * n2 + j] = dist

        # Compute summary statistics
        total_sum = 0.0
        max_distance = 0.0
        for i in range(n1):
            for j in range(n2):
                total_sum += result[i * n2 + j]
                if result[i * n2 + j] > max_distance:
                    max_distance = result[i * n2 + j]

        mid_i = n1 // 2
        mid_j = n2 // 2
        distance_at_mid_mid = result[mid_i * n2 + mid_j]

    free(d1)
    free(d2)
    free(result)

    return (total_sum, max_distance, distance_at_mid_mid)
