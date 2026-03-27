# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum arrays using fused-type memoryview parameter.

Demonstrates fused type with memoryview: a single helper
function works on both int[:] and double[:] memoryviews.

Keywords: numerical, fused type, memoryview, sum, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused numeric_t:
    int
    double


cdef double _sum_view(numeric_t[:] arr, int n):
    """Sum elements of a typed memoryview."""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += <double>arr[i]
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def fused_memview_sum(int n):
    """Sum int and double memoryviews via fused helper."""
    cdef int i

    # Int memoryview
    int_arr = cvarray(
        shape=(n,),
        itemsize=sizeof(int),
        format="i",
    )
    cdef int[:] int_view = int_arr
    for i in range(n):
        int_view[i] = (i * 37 + 13) % 997

    # Double memoryview
    dbl_arr = cvarray(
        shape=(n,),
        itemsize=sizeof(double),
        format="d",
    )
    cdef double[:] dbl_view = dbl_arr
    for i in range(n):
        dbl_view[i] = (
            <double>((i * 41 + 7) % 1009) / 17.0
        )

    cdef double int_total = _sum_view[int](
        int_view, n,
    )
    cdef double dbl_total = _sum_view[double](
        dbl_view, n,
    )
    return int_total + dbl_total
