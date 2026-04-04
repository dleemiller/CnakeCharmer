# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Bootstrap mean variance via deterministic resampling (Cython-optimized).

Keywords: statistics, bootstrap, resampling, variance, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def bootstrap_mean(int n):
    """Compute variance of k=1000 bootstrap means using C arrays.

    Args:
        n: Size of the dataset.

    Returns:
        Variance of the bootstrap means.
    """
    cdef int k = 1000
    cdef int i, b, j, idx
    cdef double total, sum_m, avg, var, diff

    cdef double *data = <double *>malloc(n * sizeof(double))
    cdef double *means = <double *>malloc(k * sizeof(double))

    if data == NULL or means == NULL:
        if data != NULL:
            free(data)
        if means != NULL:
            free(means)
        raise MemoryError("Failed to allocate bootstrap arrays")

    # Build dataset
    for i in range(n):
        data[i] = ((i * 17 + 5) % 1000) / 10.0

    # Compute bootstrap means
    for b in range(k):
        total = 0.0
        for j in range(n):
            idx = (j * b * 31 + 7) % n
            total += data[idx]
        means[b] = total / n

    # Compute variance of means
    sum_m = 0.0
    for b in range(k):
        sum_m += means[b]
    avg = sum_m / k

    var = 0.0
    for b in range(k):
        diff = means[b] - avg
        var += diff * diff
    var /= k

    free(data)
    free(means)
    return var
