# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Running factorial sum modulo a fixed prime (Cython)."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(250000, 1_000_000_007))
def factorial_mod_sum(int limit, int mod):
    return _factorial_mod_sum_impl(limit, mod)


cdef int _factorial_mod_sum_impl(int limit, int mod) noexcept:
    cdef int i
    cdef long long fact = 1
    cdef int total = 0
    for i in range(1, limit + 1):
        fact = _mul_mod(fact, i, mod)
        total += <int>fact
        if total >= mod:
            total -= mod
    return total


cdef inline long long _mul_mod(long long acc, int i, int mod) noexcept:
    return (acc * i) % mod
