# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel transform chain (scale, shift, clamp) using prange.

Keywords: numerical, transform, chain, prange, parallel, cython, benchmark
"""

from cython.parallel import prange
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def prange_transform_chain(int n):
    """Apply scale/shift/clamp with prange, return sum."""
    cdef int i
    cdef double val
    cdef double total = 0.0
    cdef unsigned int h

    for i in prange(n, nogil=True):
        h = <unsigned int>(
            <long long>i * <long long>2654435761
        ) & <unsigned int>0xFFFFFFFF
        val = h / 4294967296.0
        # scale
        val = val * 2.5
        # shift
        val = val - 0.75
        # clamp
        if val < 0.0:
            val = 0.0
        elif val > 1.5:
            val = 1.5
        total += val

    return total
