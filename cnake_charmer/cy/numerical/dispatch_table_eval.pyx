# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dispatch evaluations through array of function pointers (Cython-optimized).

Keywords: numerical, dispatch, function pointer, function table, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

ctypedef double (*eval_fn)(double) noexcept


cdef double _square(double x) noexcept:
    return x * x


cdef double _cube(double x) noexcept:
    return x * x * x


cdef double _negate(double x) noexcept:
    return -x


cdef double _half(double x) noexcept:
    return x * 0.5


@cython_benchmark(syntax="cy", args=(100000,))
def dispatch_table_eval(int n):
    """Dispatch evaluations through a C function pointer table."""
    cdef eval_fn funcs[4]
    funcs[0] = _square
    funcs[1] = _cube
    funcs[2] = _negate
    funcs[3] = _half

    cdef double total = 0.0
    cdef int i
    cdef double x
    cdef int func_idx

    for i in range(n):
        x = ((i * 37 + 11) % 997) / 100.0
        func_idx = i % 4
        total += funcs[func_idx](x)

    return total
