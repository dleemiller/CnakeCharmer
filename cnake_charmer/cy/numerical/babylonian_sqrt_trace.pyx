# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
"""Babylonian square-root iteration with convergence trace (Cython).

Sourced from SFT DuckDB blob: 32dde1ea10e5554dbd70a4b13a8fb0b220436fd9
Keywords: babylonian sqrt, newton method, convergence, numerical, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef inline double _round_to(double x, int decimals):
    cdef double scale = 1.0
    cdef int i
    cdef double y
    for i in range(decimals):
        scale *= 10.0
    if x >= 0.0:
        y = x * scale + 0.5
    else:
        y = x * scale - 0.5
    return <long long>y / scale


cdef inline double _initial_guess(double value):
    if value < 1.0:
        return 1.0
    return value * 0.5


cdef int _iterate_sqrt(double value, int loops, double tolerance, double *x_out, double *diff_out):
    cdef double x = _initial_guess(value)
    cdef double x_prev
    cdef double last_diff = 0.0
    cdef int i, used = 0

    for i in range(loops):
        x_prev = x
        x = 0.5 * (x + value / x)
        last_diff = x - x_prev
        if last_diff < 0.0:
            last_diff = -last_diff
        used = i + 1
        if last_diff < tolerance:
            break

    x_out[0] = x
    diff_out[0] = last_diff
    return used


@cython_benchmark(syntax="cy", args=(12345.6789, 1000000, 0.0))
def babylonian_sqrt_trace(double value, int loops, double tolerance):
    cdef double x = 0.0
    cdef double last_diff = 0.0
    cdef int used = 0

    if value < 0.0:
        raise ValueError("value must be non-negative")
    if value == 0.0:
        return (0.0, 0.0, 0)
    used = _iterate_sqrt(value, loops, tolerance, &x, &last_diff)

    return (_round_to(x, 12), _round_to(last_diff, 16), used)
