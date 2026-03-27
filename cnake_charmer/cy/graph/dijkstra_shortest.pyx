# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dijkstra shortest path with relaxation counting (Cython-optimized).

Keywords: graph, dijkstra, shortest path, weighted, relaxation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(4000,))
def dijkstra_shortest(int n):
    """Compute shortest paths from node 0 with relaxation counting."""
    cdef long long INF = 10**18
    cdef int i, u, v, w, best_u
    cdef long long min_d, nd
    cdef int total_relaxations = 0
    cdef int EDGES_PER_NODE = 3
    cdef int mid = n // 2

    cdef int *adj_v = <int *>malloc(n * EDGES_PER_NODE * sizeof(int))
    cdef int *adj_w = <int *>malloc(n * EDGES_PER_NODE * sizeof(int))
    cdef long long *dist = <long long *>malloc(n * sizeof(long long))
    cdef char *visited = <char *>malloc(n * sizeof(char))

    if adj_v == NULL or adj_w == NULL or dist == NULL or visited == NULL:
        if adj_v != NULL: free(adj_v)
        if adj_w != NULL: free(adj_w)
        if dist != NULL: free(dist)
        if visited != NULL: free(visited)
        raise MemoryError("Failed to allocate Dijkstra arrays")

    # Build adjacency
    for i in range(n):
        adj_v[i * 3] = (i * 5 + 3) % n
        adj_w[i * 3] = i % 8 + 1
        adj_v[i * 3 + 1] = (i * 9 + 7) % n
        adj_w[i * 3 + 1] = i % 6 + 2
        adj_v[i * 3 + 2] = (i * 13 + 11) % n
        adj_w[i * 3 + 2] = i % 4 + 3

    for i in range(n):
        dist[i] = INF
        visited[i] = 0
    dist[0] = 0

    for _ in range(n):
        best_u = -1
        min_d = INF
        for v in range(n):
            if visited[v] == 0 and dist[v] < min_d:
                min_d = dist[v]
                best_u = v
        if best_u == -1:
            break
        u = best_u
        visited[u] = 1
        for i in range(EDGES_PER_NODE):
            v = adj_v[u * 3 + i]
            w = adj_w[u * 3 + i]
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                total_relaxations += 1

    cdef long long dist_to_last = dist[n - 1] if dist[n - 1] < INF else -1
    cdef long long dist_to_mid = dist[mid] if dist[mid] < INF else -1

    free(adj_v)
    free(adj_w)
    free(dist)
    free(visited)
    return (dist_to_last, dist_to_mid, total_relaxations)
