# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Detect cycle length using Floyd's tortoise and hare algorithm.

Keywords: algorithms, floyd, cycle detection, tortoise hare, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def floyd_cycle(int n):
    """Detect cycle length in sequence f(x) = (x*x + 1) % n starting from x=2."""
    cdef long long tortoise, hare
    cdef long long mod = n
    cdef int cycle_len

    # Phase 1: find meeting point
    tortoise = (2 * 2 + 1) % mod
    hare = (tortoise * tortoise + 1) % mod
    while tortoise != hare:
        tortoise = (tortoise * tortoise + 1) % mod
        hare = (hare * hare + 1) % mod
        hare = (hare * hare + 1) % mod

    # Phase 2: find cycle length
    cycle_len = 1
    hare = (tortoise * tortoise + 1) % mod
    while tortoise != hare:
        hare = (hare * hare + 1) % mod
        cycle_len += 1

    return cycle_len
