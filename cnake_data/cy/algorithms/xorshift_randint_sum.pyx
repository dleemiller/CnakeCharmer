# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Xorshift32 random sequence checksum (Cython)."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2463534242, 500000, 1000))
def xorshift_randint_sum(unsigned int seed, int draws, int bucket):
    return _xorshift_randint_sum_impl(seed, draws, bucket)


cdef long long _xorshift_randint_sum_impl(unsigned int seed, int draws, int bucket) noexcept:
    cdef unsigned int x = seed
    cdef int i
    cdef long long total = 0
    for i in range(draws):
        x = _xorshift32_next(x)
        total += x % <unsigned int>bucket
    return total


cdef inline unsigned int _xorshift32_next(unsigned int x) noexcept:
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x
