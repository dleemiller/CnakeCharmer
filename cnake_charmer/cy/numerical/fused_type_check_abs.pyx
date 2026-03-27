# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum absolute values using fused type with type checking.

Uses `if numeric_t is int:` pattern to dispatch between
abs() for integers and fabs() for doubles.

Keywords: numerical, fused type, type check, absolute value, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark

ctypedef long long int64

ctypedef fused numeric_t:
    int
    int64
    double


cdef double _abs_sum(numeric_t *arr, int n):
    """Sum abs values with type-specific dispatch."""
    cdef double total = 0.0
    cdef int i
    if numeric_t is int or numeric_t is int64:
        for i in range(n):
            total += abs(arr[i])
    else:
        for i in range(n):
            total += fabs(arr[i])
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def fused_type_check_abs(int n):
    """Sum abs values of int, long long, double arrays."""
    cdef int i

    cdef int *int_arr = <int *>malloc(
        n * sizeof(int)
    )
    cdef int64 *ll_arr = <int64 *>malloc(
        n * sizeof(int64)
    )
    cdef double *dbl_arr = <double *>malloc(
        n * sizeof(double)
    )
    if not int_arr or not ll_arr or not dbl_arr:
        raise MemoryError()

    for i in range(n):
        int_arr[i] = (i * 37 + 13) % 997 - 498
    for i in range(n):
        ll_arr[i] = (
            <int64>((i * 53 + 29) % 10007)
            - <int64>5003
        )
    for i in range(n):
        dbl_arr[i] = (
            <double>((i * 41 + 7) % 1009 - 504)
            / 13.0
        )

    cdef double r1 = _abs_sum[int](int_arr, n)
    cdef double r2 = _abs_sum[int64](ll_arr, n)
    cdef double r3 = _abs_sum[double](dbl_arr, n)

    free(int_arr)
    free(ll_arr)
    free(dbl_arr)
    return r1 + r2 + r3
