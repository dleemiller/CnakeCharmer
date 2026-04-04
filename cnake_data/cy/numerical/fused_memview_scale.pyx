# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""In-place scale of memoryviews using fused type helper.

Demonstrates fused type with in-place memoryview mutation:
a single helper scales both float[:] and double[:].

Keywords: numerical, fused type, memoryview, scale, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_data.benchmarks import cython_benchmark

ctypedef fused num_t:
    float
    double


cdef void _scale_view(
    num_t[:] arr, int n, double factor,
):
    """Scale memoryview elements in-place."""
    cdef int i
    for i in range(n):
        arr[i] = <num_t>(arr[i] * factor)


@cython_benchmark(syntax="cy", args=(100000,))
def fused_memview_scale(int n):
    """Scale float and double memoryviews via fused helper."""
    cdef int i
    cdef double factor = 2.5
    cdef double float_total, dbl_total

    # Float memoryview
    f_arr = cvarray(
        shape=(n,),
        itemsize=sizeof(float),
        format="f",
    )
    cdef float[:] f_view = f_arr
    for i in range(n):
        f_view[i] = <float>(
            ((i * 31 + 17) % 503) / 11.0
        )
    _scale_view[float](f_view, n, factor)

    float_total = 0.0
    for i in range(n):
        float_total += <double>f_view[i]

    # Double memoryview
    d_arr = cvarray(
        shape=(n,),
        itemsize=sizeof(double),
        format="d",
    )
    cdef double[:] d_view = d_arr
    for i in range(n):
        d_view[i] = ((i * 43 + 19) % 607) / 13.0
    _scale_view[double](d_view, n, factor)

    dbl_total = 0.0
    for i in range(n):
        dbl_total += d_view[i]

    return float_total + dbl_total
