# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Huber loss between two deterministic sequences (Cython-optimized).

Keywords: statistics, huber, loss, robust, regression, cython, benchmark
"""

from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def huber_loss(int n):
    """Compute Huber loss between two deterministic sequences of length n.

    Args:
        n: Length of the sequences.

    Returns:
        Tuple of (average huber loss, max absolute residual).
    """
    cdef double delta = 0.0125
    cdef double res_sum = 0.0
    cdef double max_res = 0.0
    cdef double res
    cdef int i
    cdef double* x = <double*>malloc(n * sizeof(double))
    cdef double* y = <double*>malloc(n * sizeof(double))

    if x is NULL or y is NULL:
        if x is not NULL:
            free(x)
        if y is not NULL:
            free(y)
        raise MemoryError("Failed to allocate arrays")

    # Pre-generate sequences
    for i in range(n):
        x[i] = ((i * 7 + 3) % 1000) / 500.0 - 1.0
        y[i] = ((i * 13 + 7) % 1000) / 500.0 - 1.0

    with nogil:
        for i in range(n):
            res = fabs(x[i] - y[i])

            if res > max_res:
                max_res = res

            if res <= delta:
                res_sum += (res * res) / 2.0
            else:
                res_sum += delta * (res - delta / 2.0)

    free(x)
    free(y)

    return (res_sum / n, max_res)
