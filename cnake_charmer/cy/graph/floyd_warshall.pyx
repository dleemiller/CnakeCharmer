# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Floyd-Warshall all-pairs shortest paths (Cython-optimized).

Keywords: graph, floyd-warshall, shortest path, all-pairs, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def floyd_warshall(int n):
    """Compute all-pairs shortest paths on n nodes.

    Args:
        n: Number of nodes.

    Returns:
        Sum of all finite shortest-path distances.
    """
    cdef int INF = 1000000000
    cdef int i, j, k, w, jj
    cdef int ik, new_dist
    cdef long long total

    cdef int *dist = <int *>malloc(n * n * sizeof(int))
    if not dist:
        raise MemoryError()

    # Initialize
    for i in range(n * n):
        dist[i] = INF
    for i in range(n):
        dist[i * n + i] = 0
        jj = (i * 3 + 1) % n
        w = (i % 10) + 1
        if w < dist[i * n + jj]:
            dist[i * n + jj] = w

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            ik = dist[i * n + k]
            if ik == INF:
                continue
            for j in range(n):
                new_dist = ik + dist[k * n + j]
                if new_dist < dist[i * n + j]:
                    dist[i * n + j] = new_dist

    total = 0
    for i in range(n * n):
        if dist[i] < INF:
            total += dist[i]

    free(dist)
    return total
