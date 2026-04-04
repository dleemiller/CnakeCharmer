# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sin-squared accumulation over a synthetic grid (Cython)."""

from libc.math cimport sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(0.0, 0.001, 50000))
def sin_squared_sum(double offset, double step, int samples):
    return _sin_squared_sum_impl(offset, step, samples)


cdef double _sin_squared_sum_impl(double offset, double step, int samples) noexcept:
    cdef int i
    cdef double x, s
    cdef double total = 0.0
    for i in range(samples):
        x = offset + i * step
        s = _sin_sq(x)
        total += s
    return total


cdef inline double _sin_sq(double x) noexcept:
    cdef double s = sin(x)
    return s * s
