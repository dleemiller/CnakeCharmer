# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find strongest short-lag cyclic autocorrelation in an integer signal (Cython)."""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef void _autocorr_kernel(
    int* seq,
    int n,
    int* best_lag_out,
    long long* best_val_out,
    long long* lag1_val_out,
) noexcept nogil:
    cdef int i, lag
    cdef long long total
    cdef int best_lag = 1
    cdef long long best_val = -1000000000000000000
    cdef long long lag1_val = 0

    for lag in range(1, 17):
        total = 0
        for i in range(n):
            total += seq[i] * seq[(i + lag) % n]
        if lag == 1:
            lag1_val = total
        if total > best_val:
            best_val = total
            best_lag = lag

    best_lag_out[0] = best_lag
    best_val_out[0] = best_val
    lag1_val_out[0] = lag1_val


@cython_benchmark(syntax="cy", args=(8000,))
def cyclic_autocorr_peak(int n):
    cdef int *seq = <int *>malloc(n * sizeof(int))
    cdef int i
    cdef int best_lag = 1
    cdef long long best_val = -1000000000000000000
    cdef long long lag1_val = 0

    if not seq:
        raise MemoryError()

    for i in range(n):
        seq[i] = ((i * 29 + 17) % 31) - 15

    with nogil:
        _autocorr_kernel(seq, n, &best_lag, &best_val, &lag1_val)

    free(seq)
    return (best_lag, best_val, lag1_val)
