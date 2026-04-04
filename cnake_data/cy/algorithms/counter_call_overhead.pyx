# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
"""Compare direct accumulation vs helper-call and generator-call styles (Cython).

Sourced from SFT DuckDB blob: dd6f108767d588dd0632ce9cb0d07b1da90c822a
Keywords: loops, helper function, generator, call overhead, algorithms, cython
"""

from cnake_data.benchmarks import cython_benchmark


cdef int _helper(int v):
    return v


cdef inline long long _direct_sum(int limit, int offset):
    cdef int i
    cdef long long acc = 0
    for i in range(limit):
        acc += i + offset
    return acc


cdef inline long long _helper_sum(int limit, int offset):
    cdef int i
    cdef long long acc = 0
    for i in range(limit):
        acc += _helper(i + offset)
    return acc


cdef inline long long _helper_gen_sum(int limit, int offset):
    cdef int i
    cdef long long acc = 0
    for i in range(limit):
        acc += i + offset
    return acc


cdef void _counter_core(int limit, int repeats, int offset, long long *direct, long long *via_helper, long long *via_gen):
    cdef int r
    cdef long long d = 0
    cdef long long h = 0
    cdef long long g = 0
    for r in range(repeats):
        d += _direct_sum(limit, offset)
        h += _helper_sum(limit, offset)
        g += _helper_gen_sum(limit, offset)
    direct[0] = d
    via_helper[0] = h
    via_gen[0] = g


@cython_benchmark(syntax="cy", args=(200000, 4, 3))
def counter_call_overhead(int limit, int repeats, int offset):
    cdef long long direct = 0
    cdef long long via_helper = 0
    cdef long long via_gen = 0

    _counter_core(limit, repeats, offset, &direct, &via_helper, &via_gen)
    return (direct, via_helper, via_gen)
