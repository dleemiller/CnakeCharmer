# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bellman-Ford shortest paths with negative edge detection (Cython-optimized).

Keywords: graph, bellman-ford, shortest path, negative cycle, relaxation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def bellman_ford_paths(int n):
    """Compute shortest paths from node 0 using Bellman-Ford with C arrays."""
    cdef long long INF = 10**18
    cdef int i, e, iteration
    cdef int u, v, w
    cdef long long nd
    cdef int updated
    cdef int num_edges = n * 3
    cdef int mid = n // 2
    cdef int negative_cycle_found = 0

    cdef int *edges_u = <int *>malloc(num_edges * sizeof(int))
    cdef int *edges_v = <int *>malloc(num_edges * sizeof(int))
    cdef int *edges_w = <int *>malloc(num_edges * sizeof(int))
    cdef long long *dist = <long long *>malloc(n * sizeof(long long))

    if not edges_u or not edges_v or not edges_w or not dist:
        free(edges_u); free(edges_v); free(edges_w); free(dist)
        raise MemoryError()

    # Build edge list
    for i in range(n):
        edges_u[i * 3] = i
        edges_v[i * 3] = (i * 3 + 1) % n
        edges_w[i * 3] = (i % 10) + 1

        edges_u[i * 3 + 1] = i
        edges_v[i * 3 + 1] = (i * 7 + 2) % n
        edges_w[i * 3 + 1] = (i % 5) + 2

        edges_u[i * 3 + 2] = i
        edges_v[i * 3 + 2] = (i * 11 + 5) % n
        edges_w[i * 3 + 2] = (i % 4) + 1

    for i in range(n):
        dist[i] = INF
    dist[0] = 0

    for iteration in range(n - 1):
        updated = 0
        for e in range(num_edges):
            u = edges_u[e]
            v = edges_v[e]
            w = edges_w[e]
            if dist[u] < INF:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    updated = 1
        if updated == 0:
            break

    # Check for negative cycles
    for e in range(num_edges):
        u = edges_u[e]
        v = edges_v[e]
        w = edges_w[e]
        if dist[u] < INF and dist[u] + w < dist[v]:
            negative_cycle_found = 1
            break

    cdef long long dist_to_last = dist[n - 1] if dist[n - 1] < INF else -1
    cdef long long dist_to_mid = dist[mid] if dist[mid] < INF else -1

    free(edges_u)
    free(edges_v)
    free(edges_w)
    free(dist)
    return (dist_to_last, negative_cycle_found, dist_to_mid)
