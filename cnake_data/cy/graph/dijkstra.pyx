# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Dijkstra's shortest path from node 0 on a weighted deterministic graph (Cython-optimized).

Keywords: graph, dijkstra, shortest path, weighted, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def dijkstra(int n):
    """Compute sum of shortest path distances from node 0 using O(V^2) Dijkstra with C arrays.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (total distance sum, max distance, reachable node count).
    """
    cdef long long INF = 10**18
    cdef int i, u, v, w, best_u
    cdef long long min_d, nd
    cdef int EDGES_PER_NODE = 3

    # Flat adjacency: node i has edges at adj_v[i*3..i*3+2], adj_w[i*3..i*3+2]
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
        adj_v[i * 3] = (i * 3 + 1) % n
        adj_w[i * 3] = i % 10 + 1
        adj_v[i * 3 + 1] = (i * 7 + 2) % n
        adj_w[i * 3 + 1] = i % 5 + 1
        adj_v[i * 3 + 2] = (i * 11 + 3) % n
        adj_w[i * 3 + 2] = i % 7 + 1

    # Initialize
    for i in range(n):
        dist[i] = INF
        visited[i] = 0
    dist[0] = 0

    # O(V^2) Dijkstra
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

    cdef long long total = 0
    cdef long long max_dist = 0
    cdef int reachable_count = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            reachable_count += 1

    free(adj_v)
    free(adj_w)
    free(dist)
    free(visited)
    return (total, max_dist, reachable_count)
