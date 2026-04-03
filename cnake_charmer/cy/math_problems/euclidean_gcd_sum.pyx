# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of Euclidean GCD over synthetic integer pairs (Cython)."""

from cnake_charmer.benchmarks import cython_benchmark


cdef inline int _gcd(int a, int b) noexcept:
    cdef int t
    while b:
        t = a % b
        a = b
        b = t
    return a


@cython_benchmark(syntax="cy", args=(6789, 9876, 120000))
def euclidean_gcd_sum(int seed_a, int seed_b, int count):
    return _euclidean_gcd_sum_impl(seed_a, seed_b, count)


cdef int _euclidean_gcd_sum_impl(int seed_a, int seed_b, int count) noexcept:
    cdef int total = 0
    cdef int i
    cdef int a = seed_a
    cdef int b = seed_b
    for i in range(count):
        a = (a * 1664525 + 1013904223) & 0x7FFFFFFF
        b = (b * 1103515245 + 12345) & 0x7FFFFFFF
        total += _gcd(a, b)
    return total
