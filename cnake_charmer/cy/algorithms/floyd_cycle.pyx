# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Detect cycle lengths using Floyd's tortoise and hare algorithm.

Keywords: algorithms, floyd, cycle detection, tortoise hare, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def floyd_cycle(int n):
    """Sum cycle lengths for n different sequences f(x) = (x*x + c) % 1000003."""
    cdef long long tortoise, hare, c
    cdef long long mod = 1000003
    cdef int cycle_len, i
    cdef long long total = 0

    for i in range(n):
        c = i + 1
        tortoise = (2 * 2 + c) % mod
        hare = (tortoise * tortoise + c) % mod
        while tortoise != hare:
            tortoise = (tortoise * tortoise + c) % mod
            hare = (hare * hare + c) % mod
            hare = (hare * hare + c) % mod

        cycle_len = 1
        hare = (tortoise * tortoise + c) % mod
        while tortoise != hare:
            hare = (hare * hare + c) % mod
            cycle_len += 1

        total += cycle_len

    return int(total)
