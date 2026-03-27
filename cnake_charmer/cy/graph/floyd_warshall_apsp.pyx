# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Floyd-Warshall all-pairs shortest paths with path counting (Cython-optimized).

Keywords: graph, floyd-warshall, all-pairs, shortest path, path counting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(400,))
def floyd_warshall_apsp(int n):
    """Compute all-pairs shortest paths and count finite paths."""
    cdef int INF = 1000000000
    cdef int i, j, k, w0, w1, w2, j0, j1, j2
    cdef int ik, new_dist
    cdef int mid = n // 2
    cdef long long total_finite_paths = 0

    cdef int *dist = <int *>malloc(n * n * sizeof(int))
    if not dist:
        raise MemoryError()

    # Initialize
    for i in range(n * n):
        dist[i] = INF
    for i in range(n):
        dist[i * n + i] = 0
        # Edge to next node (ensures connectivity)
        j0 = (i + 1) % n
        w0 = (i % 5) + 1
        if w0 < dist[i * n + j0]:
            dist[i * n + j0] = w0
        j1 = (i * 3 + 1) % n
        w1 = (i % 7) + 2
        if w1 < dist[i * n + j1]:
            dist[i * n + j1] = w1
        j2 = (i * 5 + 2) % n
        w2 = (i % 4) + 3
        if w2 < dist[i * n + j2]:
            dist[i * n + j2] = w2

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

    cdef int dist_0_to_last = dist[0 * n + n - 1]
    cdef int dist_mid_to_last = dist[mid * n + n - 1]

    for i in range(n * n):
        if dist[i] < INF:
            total_finite_paths += 1

    free(dist)
    return (dist_0_to_last, dist_mid_to_last, total_finite_paths)
