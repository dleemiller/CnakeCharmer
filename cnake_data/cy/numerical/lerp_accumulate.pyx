# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Linear interpolation accumulation (Cython)."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1.25, 5.5, 250000, 0.00001))
def lerp_accumulate(double a0, double b0, int steps, double delta):
    return _lerp_accumulate_impl(a0, b0, steps, delta)


cdef double _lerp_accumulate_impl(double a0, double b0, int steps, double delta) noexcept:
    cdef int i
    cdef double total = 0.0
    cdef double t
    cdef double a = a0
    cdef double b = b0
    for i in range(steps):
        t = (i % 1024) * 0.0009765625
        total += a + t * (b - a)
        a += delta
        b -= delta * 0.5
    return total
