# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bellman-Ford single-source shortest path from node 0 (Cython-optimized).

Keywords: graph, bellman-ford, shortest path, weighted, relaxation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def bellman_ford(int n):
    """Compute sum of shortest path distances from node 0 using Bellman-Ford with C arrays."""
    cdef long long INF = 10**18
    cdef int i, u, v, w, iteration
    cdef long long nd
    cdef int num_edges = n * 2
    cdef int updated

    cdef int *edge_u = <int *>malloc(num_edges * sizeof(int))
    cdef int *edge_v = <int *>malloc(num_edges * sizeof(int))
    cdef int *edge_w = <int *>malloc(num_edges * sizeof(int))
    cdef long long *dist = <long long *>malloc(n * sizeof(long long))

    if not edge_u or not edge_v or not edge_w or not dist:
        free(edge_u); free(edge_v); free(edge_w); free(dist)
        raise MemoryError()

    # Build edge list
    for i in range(n):
        edge_u[i * 2] = i
        edge_v[i * 2] = (i * 3 + 1) % n
        edge_w[i * 2] = i % 10 + 1
        edge_u[i * 2 + 1] = i
        edge_v[i * 2 + 1] = (i * 7 + 2) % n
        edge_w[i * 2 + 1] = i % 5 + 1

    for i in range(n):
        dist[i] = INF
    dist[0] = 0

    for iteration in range(n - 1):
        updated = 0
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            w = edge_w[i]
            if dist[u] < INF:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    updated = 1
        if updated == 0:
            break

    cdef long long total = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]

    free(edge_u)
    free(edge_v)
    free(edge_w)
    free(dist)
    return total
