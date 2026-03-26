# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Shortest path in a DAG using topological ordering (Cython-optimized).

Keywords: graph, DAG, shortest path, topological sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def shortest_path_dag(int n):
    """Compute shortest paths from node 0 in a weighted DAG using C arrays."""
    if n < 1:
        return (0, 0, 0)

    cdef long long INF = 1000000000000000000LL  # 10^18
    cdef int i, t1, t2, w1, w2
    cdef int edge_count = 0

    # Count edges first
    for i in range(n):
        t1 = i + 1 + i % 3
        if t1 < n:
            edge_count += 1
        t2 = i + 2 + i % 5
        if t2 < n:
            edge_count += 1

    # Build CSR adjacency for forward edges
    cdef int *adj_target = <int *>malloc(edge_count * sizeof(int))
    cdef int *adj_weight = <int *>malloc(edge_count * sizeof(int))
    cdef int *adj_offset = <int *>malloc((n + 1) * sizeof(int))
    cdef long long *dist = <long long *>malloc(n * sizeof(long long))

    if not adj_target or not adj_weight or not adj_offset or not dist:
        if adj_target: free(adj_target)
        if adj_weight: free(adj_weight)
        if adj_offset: free(adj_offset)
        if dist: free(dist)
        raise MemoryError()

    # Count out-degree for each node
    cdef int *out_degree = <int *>malloc(n * sizeof(int))
    if not out_degree:
        free(adj_target); free(adj_weight); free(adj_offset); free(dist)
        raise MemoryError()

    memset(out_degree, 0, n * sizeof(int))
    for i in range(n):
        t1 = i + 1 + i % 3
        if t1 < n:
            out_degree[i] += 1
        t2 = i + 2 + i % 5
        if t2 < n:
            out_degree[i] += 1

    adj_offset[0] = 0
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + out_degree[i]

    # Fill adjacency
    cdef int *pos = <int *>malloc(n * sizeof(int))
    if not pos:
        free(adj_target); free(adj_weight); free(adj_offset); free(dist); free(out_degree)
        raise MemoryError()

    for i in range(n):
        pos[i] = adj_offset[i]

    for i in range(n):
        t1 = i + 1 + i % 3
        if t1 < n:
            w1 = ((i * 7 + 3) % 50) + 1
            adj_target[pos[i]] = t1
            adj_weight[pos[i]] = w1
            pos[i] += 1
        t2 = i + 2 + i % 5
        if t2 < n:
            w2 = ((i * 11 + 1) % 40) + 1
            adj_target[pos[i]] = t2
            adj_weight[pos[i]] = w2
            pos[i] += 1

    # Initialize distances
    for i in range(n):
        dist[i] = INF
    dist[0] = 0

    # Relax in topological order (0, 1, 2, ..., n-1)
    cdef int u, v, idx
    cdef long long nd
    for u in range(n):
        if dist[u] == INF:
            continue
        for idx in range(adj_offset[u], adj_offset[u + 1]):
            v = adj_target[idx]
            nd = dist[u] + adj_weight[idx]
            if nd < dist[v]:
                dist[v] = nd

    cdef long long total = 0
    cdef long long max_dist = 0
    cdef int reachable = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            reachable += 1

    free(adj_target)
    free(adj_weight)
    free(adj_offset)
    free(dist)
    free(out_degree)
    free(pos)

    return (total, max_dist, reachable)
