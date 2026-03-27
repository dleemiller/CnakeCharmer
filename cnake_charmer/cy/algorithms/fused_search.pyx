# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generic binary search using fused type.

A single fused-type binary search helper works on both
int* and double* sorted arrays.

Keywords: algorithms, binary search, fused type, generic, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused orderable_t:
    int
    double


cdef int _binary_search(
    orderable_t *arr, int n, orderable_t target,
) noexcept:
    """Binary search, return 1 if found else 0."""
    cdef int lo = 0
    cdef int hi = n - 1
    cdef int mid
    while lo <= hi:
        mid = (lo + hi) / 2
        if arr[mid] < target:
            lo = mid + 1
        elif arr[mid] > target:
            hi = mid - 1
        else:
            return 1
    return 0


@cython_benchmark(syntax="cy", args=(100000,))
def fused_search(int n):
    """Binary search on int and double sorted arrays."""
    cdef int i, q, found
    cdef int num_queries = n / 3

    cdef int *int_arr = <int *>malloc(
        n * sizeof(int)
    )
    cdef double *dbl_arr = <double *>malloc(
        n * sizeof(double)
    )
    if not int_arr or not dbl_arr:
        raise MemoryError()

    for i in range(n):
        int_arr[i] = i * 3 + 1
    for i in range(n):
        dbl_arr[i] = i * 2.7 + 0.5

    found = 0

    cdef int target_i
    for q in range(num_queries):
        target_i = (q * 7 + 3) % (n * 3)
        found += _binary_search[int](
            int_arr, n, target_i,
        )

    cdef double target_d
    for q in range(num_queries):
        target_d = (
            <double>((q * 11 + 5) % n) * 2.7 + 0.5
        )
        found += _binary_search[double](
            dbl_arr, n, target_d,
        )

    free(int_arr)
    free(dbl_arr)
    return found
