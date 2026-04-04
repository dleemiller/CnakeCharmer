# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Clamp hash-derived values to a range and sum results
(Cython-optimized with cpdef standalone function).

Keywords: clamp, numerical, cpdef, standalone function, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cpdef double clamp(double val, double lo, double hi):
    """Clamp val to the range [lo, hi]."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


@cython_benchmark(syntax="cy", args=(100000,))
def cpdef_clamp_sum(int n):
    """Clamp n hash-derived values to [-5.0, 5.0], sum."""
    cdef double total = 0.0
    cdef int i
    cdef unsigned long long h
    cdef double val

    for i in range(n):
        h = (<unsigned long long>i
             * <unsigned long long>2654435761)
        val = (
            <double>(h & <unsigned long long>0xFFFFFFFF)
            / 4294967295.0 * 20.0 - 10.0
        )
        total += clamp(val, -5.0, 5.0)
    return total
