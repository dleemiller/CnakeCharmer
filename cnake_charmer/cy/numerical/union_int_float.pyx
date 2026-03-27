# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Reinterpret integer bits as float via union type punning
and sum finite results (Cython-optimized).

Keywords: union, type punning, int, float, numerical, cython, benchmark
"""

from libc.math cimport isfinite

from cnake_charmer.benchmarks import cython_benchmark

cdef union IntFloat:
    unsigned int i
    float f


@cython_benchmark(syntax="cy", args=(100000,))
def union_int_float(int n):
    """Reinterpret n hash-derived ints as floats, sum finite."""
    cdef double total = 0.0
    cdef int idx
    cdef unsigned long long h
    cdef IntFloat pun

    for idx in range(n):
        h = (<unsigned long long>idx
             * <unsigned long long>2654435761)
        pun.i = <unsigned int>(h & <unsigned long long>0xFFFFFFFF)
        if isfinite(pun.f):
            total += pun.f
    return total
