# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Descriptive statistics with struct return.

Returns DescStats struct with mean, variance, skewness.
Demonstrates multi-field struct return from cdef helper.

Keywords: statistics, descriptive, struct return, skewness, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef struct DescStats:
    double mean
    double variance
    double skewness


cdef DescStats _compute_desc_stats(
    double *arr, int n,
) noexcept:
    """Compute mean, variance, skewness."""
    cdef DescStats st
    cdef int i
    cdef double s = 0.0
    cdef double var_sum = 0.0
    cdef double skew_sum = 0.0
    cdef double diff, std_val

    for i in range(n):
        s += arr[i]
    st.mean = s / <double>n

    for i in range(n):
        diff = arr[i] - st.mean
        var_sum += diff * diff
        skew_sum += diff * diff * diff

    st.variance = var_sum / <double>n
    if st.variance > 0.0:
        std_val = sqrt(st.variance)
        st.skewness = (
            (skew_sum / <double>n)
            / (std_val * std_val * std_val)
        )
    else:
        st.skewness = 0.0
    return st


@cython_benchmark(syntax="cy", args=(100000,))
def struct_return_stats(int n):
    """Compute descriptive stats, return weighted sum."""
    cdef int i
    cdef unsigned int h

    cdef double *arr = <double *>malloc(
        n * sizeof(double)
    )
    if not arr:
        raise MemoryError()

    for i in range(n):
        h = (
            (<unsigned int>i
             * <unsigned int>2654435761)
            ^ (<unsigned int>i
               * <unsigned int>2246822519)
        )
        arr[i] = <double>(h & 0xFFFF) / 65535.0

    cdef DescStats st = _compute_desc_stats(arr, n)

    free(arr)
    return st.mean + st.variance + fabs(st.skewness)
