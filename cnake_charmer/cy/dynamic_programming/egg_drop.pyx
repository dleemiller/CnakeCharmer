# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Minimum trials for egg drop problem using full DP table (Cython-optimized).

Keywords: dynamic programming, egg drop, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def egg_drop(int n):
    """Compute minimum trials with 3 eggs and n floors using C arrays."""
    cdef int eggs = 3
    cdef int e, f, x, val, best
    cdef int opt_x
    cdef int *prev_egg
    cdef int *cur_egg
    cdef int result

    if n <= 0:
        return 0
    if n == 1:
        return 1

    prev_egg = <int *>malloc((n + 1) * sizeof(int))
    cur_egg = <int *>malloc((n + 1) * sizeof(int))
    if not prev_egg or not cur_egg:
        raise MemoryError()

    # e=1: dp[1][f] = f
    for f in range(n + 1):
        prev_egg[f] = f

    for e in range(2, eggs + 1):
        cur_egg[0] = 0
        opt_x = 1
        for f in range(1, n + 1):
            best = n + 1
            for x in range(opt_x, f + 1):
                if prev_egg[x - 1] > cur_egg[f - x]:
                    val = 1 + prev_egg[x - 1]
                else:
                    val = 1 + cur_egg[f - x]
                if val < best:
                    best = val
                    opt_x = x
                elif prev_egg[x - 1] > cur_egg[f - x]:
                    break
            cur_egg[f] = best
        # Swap
        for f in range(n + 1):
            prev_egg[f] = cur_egg[f]

    result = prev_egg[n]
    free(prev_egg)
    free(cur_egg)
    return result
