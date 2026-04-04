# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply different cdef callback transformations to array (Cython-optimized).

Keywords: numerical, callback, transform, function pointer, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_data.benchmarks import cython_benchmark

ctypedef double (*transform_fn)(double) noexcept


cdef double _scale_up(double x) noexcept:
    return x * 2.5


cdef double _invert(double x) noexcept:
    return 1.0 / (1.0 + x * x)


cdef double _smooth(double x) noexcept:
    return x / (1.0 + fabs(x))


cdef double _transform_array(double *arr, int n, transform_fn func) noexcept:
    cdef int i
    cdef double total = 0.0
    for i in range(n):
        total += func(arr[i])
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def callback_transform(int n):
    """Apply three cdef callbacks via function pointer to array."""
    cdef double *arr = <double *>malloc(n * sizeof(double))
    if not arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        arr[i] = ((i * 43 + 7) % 503) / 50.0

    cdef double total = 0.0
    total += _transform_array(arr, n, _scale_up)
    total += _transform_array(arr, n, _invert)
    total += _transform_array(arr, n, _smooth)

    free(arr)
    return total
