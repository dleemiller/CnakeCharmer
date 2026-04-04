# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Prim's minimum spanning tree on a weighted graph (Cython-optimized).

Keywords: graph, prim, minimum spanning tree, mst, greedy, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def prim_mst(int n):
    """Compute MST of n-node weighted graph using Prim's algorithm with C arrays."""
    cdef long long INF = 10 ** 18
    cdef int i, j, u, v
    cdef int w, t
    cdef long long min_key_val
    cdef long long total_weight = 0
    cdef long long max_edge = 0
    cdef long long min_edge = INF

    # Build adjacency using edge arrays
    # Each node has up to 5 edges, but edges are bidirectional so up to 10 per node
    # Use CSR-like format: for each node store offset and edges
    # First count edges per node
    cdef int max_edges_per_node = 20  # generous upper bound
    cdef int *adj_count = <int *>malloc(n * sizeof(int))
    cdef int *adj_target = <int *>malloc(n * max_edges_per_node * sizeof(int))
    cdef int *adj_weight = <int *>malloc(n * max_edges_per_node * sizeof(int))

    cdef int *in_mst = <int *>malloc(n * sizeof(int))
    cdef long long *key = <long long *>malloc(n * sizeof(long long))
    cdef int *edge_from = <int *>malloc(n * sizeof(int))

    if not adj_count or not adj_target or not adj_weight or not in_mst or not key or not edge_from:
        free(adj_count); free(adj_target); free(adj_weight)
        free(in_mst); free(key); free(edge_from)
        raise MemoryError()

    for i in range(n):
        adj_count[i] = 0
        in_mst[i] = 0
        key[i] = INF
        edge_from[i] = -1

    # Add edges (bidirectional)
    cdef int targets[5]
    for i in range(n):
        targets[0] = (i + 1) % n
        targets[1] = (i + 3) % n
        targets[2] = (i * 7 + 2) % n
        targets[3] = (i * 13 + 5) % n
        targets[4] = (i * 31 + 11) % n

        for j in range(5):
            t = targets[j]
            if t != i:
                w = ((i * 17 + t * 31 + 7) % 997) + 1

                # Add i -> t
                if adj_count[i] < max_edges_per_node:
                    adj_target[i * max_edges_per_node + adj_count[i]] = t
                    adj_weight[i * max_edges_per_node + adj_count[i]] = w
                    adj_count[i] += 1

                # Add t -> i
                if adj_count[t] < max_edges_per_node:
                    adj_target[t * max_edges_per_node + adj_count[t]] = i
                    adj_weight[t * max_edges_per_node + adj_count[t]] = w
                    adj_count[t] += 1

    key[0] = 0

    for _ in range(n):
        # Find minimum key vertex not in MST
        u = -1
        min_key_val = INF
        for v in range(n):
            if in_mst[v] == 0 and key[v] < min_key_val:
                min_key_val = key[v]
                u = v

        if u == -1:
            break

        in_mst[u] = 1

        if edge_from[u] != -1:
            total_weight += key[u]
            if key[u] > max_edge:
                max_edge = key[u]
            if key[u] < min_edge:
                min_edge = key[u]

        # Update keys
        for j in range(adj_count[u]):
            v = adj_target[u * max_edges_per_node + j]
            w = adj_weight[u * max_edges_per_node + j]
            if in_mst[v] == 0 and w < key[v]:
                key[v] = w
                edge_from[v] = u

    if min_edge == INF:
        min_edge = 0

    free(adj_count)
    free(adj_target)
    free(adj_weight)
    free(in_mst)
    free(key)
    free(edge_from)

    return (total_weight, max_edge, min_edge)
