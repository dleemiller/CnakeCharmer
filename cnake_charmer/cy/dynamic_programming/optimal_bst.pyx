# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Optimal binary search tree cost via dynamic programming (Cython-optimized).

Keywords: optimal BST, binary search tree, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(250,))
def optimal_bst(int n):
    """Compute minimum cost of optimal BST using flat C arrays."""
    cdef int *freq = <int *>malloc(n * sizeof(int))
    cdef long long *cost = <long long *>malloc(n * n * sizeof(long long))
    cdef int *prefix = <int *>malloc((n + 1) * sizeof(int))

    if not freq or not cost or not prefix:
        if freq: free(freq)
        if cost: free(cost)
        if prefix: free(prefix)
        raise MemoryError()

    cdef int i, j, r, length, weight
    cdef long long best, left_cost, right_cost, val

    for i in range(n):
        freq[i] = (i * 7 + 3) % 100 + 1

    prefix[0] = 0
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    # Initialize cost to 0
    for i in range(n * n):
        cost[i] = 0

    # Base case
    for i in range(n):
        cost[i * n + i] = freq[i]

    # Fill for lengths 2..n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            weight = prefix[j + 1] - prefix[i]
            best = 2000000000000000000LL
            for r in range(i, j + 1):
                if r > i:
                    left_cost = cost[i * n + (r - 1)]
                else:
                    left_cost = 0
                if r < j:
                    right_cost = cost[(r + 1) * n + j]
                else:
                    right_cost = 0
                val = left_cost + right_cost
                if val < best:
                    best = val
            cost[i * n + j] = best + weight

    cdef long long result = cost[0 * n + (n - 1)]

    free(freq)
    free(cost)
    free(prefix)
    return result
