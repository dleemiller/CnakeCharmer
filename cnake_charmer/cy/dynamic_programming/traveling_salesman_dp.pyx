# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Traveling salesman problem using bitmask DP (Cython-optimized).

Keywords: dynamic programming, tsp, traveling salesman, bitmask, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(16,))
def traveling_salesman_dp(int n):
    """Solve TSP with bitmask DP using C arrays."""
    cdef int i, j, u, v, mask, new_mask
    cdef int full_mask
    cdef int *dist
    cdef int *dp
    cdef int cost, result
    cdef int INF = 1000000000

    if n <= 1:
        return 0

    # Build distance matrix (flat array)
    dist = <int *>malloc(n * n * sizeof(int))
    if not dist:
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            dist[i * n + j] = <int>(
                fabs(sin(i * 0.7) - sin(j * 0.7)) * 100.0
                + fabs(cos(i * 1.3) - cos(j * 1.3)) * 100.0
            )

    # DP table: dp[mask * n + city]
    full_mask = (1 << n) - 1
    dp = <int *>malloc((full_mask + 1) * n * sizeof(int))
    if not dp:
        free(dist)
        raise MemoryError()

    for i in range((full_mask + 1) * n):
        dp[i] = INF

    dp[1 * n + 0] = 0  # Start at city 0, mask=1

    for mask in range(1, full_mask + 1):
        for u in range(n):
            if dp[mask * n + u] >= INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                cost = dp[mask * n + u] + dist[u * n + v]
                if cost < dp[new_mask * n + v]:
                    dp[new_mask * n + v] = cost

    result = INF
    for u in range(n):
        cost = dp[full_mask * n + u] + dist[u * n + 0]
        if cost < result:
            result = cost

    free(dist)
    free(dp)
    return result
