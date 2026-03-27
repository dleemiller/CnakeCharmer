# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute GCD of hash-derived pairs and sum all GCDs
(Cython-optimized with cpdef standalone function).

Keywords: gcd, math, cpdef, standalone function, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cpdef long long gcd(long long a, long long b):
    """Compute greatest common divisor of a and b."""
    cdef long long temp
    while b:
        temp = b
        b = a % b
        a = temp
    return a


@cython_benchmark(syntax="cy", args=(100000,))
def cpdef_gcd_sum(int n):
    """Compute GCD of n hash-derived pairs, sum all GCDs."""
    cdef long long total = 0
    cdef int i
    cdef unsigned long long h1, h2
    cdef long long a, b

    for i in range(n):
        h1 = (<unsigned long long>i
              * <unsigned long long>2654435761)
        h2 = (<unsigned long long>i
              * <unsigned long long>2246822519)
        a = <long long>(
            (h1 & <unsigned long long>0xFFFFFFFF)
            % 100000 + 1
        )
        b = <long long>(
            (h2 & <unsigned long long>0xFFFFFFFF)
            % 100000 + 1
        )
        total += gcd(a, b)
    return total
