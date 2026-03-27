# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Kahn's algorithm for topological sort on a DAG (Cython-optimized).

Keywords: graph, topological sort, kahn, dag, bfs, ordering, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def topological_sort_kahn(int n):
    """Topological sort of n-node DAG using Kahn's algorithm with C arrays."""
    cdef int max_edges_per_node = 3
    cdef int *adj = <int *>malloc(n * max_edges_per_node * sizeof(int))
    cdef int *adj_count = <int *>malloc(n * sizeof(int))
    cdef int *in_degree = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))
    cdef int *order = <int *>malloc(n * sizeof(int))

    if not adj or not adj_count or not in_degree or not queue or not order:
        free(adj); free(adj_count); free(in_degree); free(queue); free(order)
        raise MemoryError()

    cdef int i, t, t1, t2, t3, u, v, j
    cdef int head = 0, tail = 0, idx = 0
    cdef int mid
    cdef int limit

    for i in range(n):
        adj_count[i] = 0
        in_degree[i] = 0

    # Build DAG edges
    for i in range(n):
        if i + 1 >= n:
            continue

        limit = n - i
        if limit > 20:
            limit = 20
        t1 = i + 1 + (i * 7 + 3) % limit

        limit = n - i
        if limit > 15:
            limit = 15
        t2 = i + 1 + (i * 13 + 7) % limit

        limit = n - i
        if limit > 10:
            limit = 10
        t3 = i + 1 + (i * 31 + 11) % limit

        # Add unique edges
        if 0 <= t1 < n and t1 != i:
            adj[i * max_edges_per_node + adj_count[i]] = t1
            adj_count[i] += 1
            in_degree[t1] += 1

        if 0 <= t2 < n and t2 != i and t2 != t1:
            adj[i * max_edges_per_node + adj_count[i]] = t2
            adj_count[i] += 1
            in_degree[t2] += 1

        if 0 <= t3 < n and t3 != i and t3 != t1 and t3 != t2:
            adj[i * max_edges_per_node + adj_count[i]] = t3
            adj_count[i] += 1
            in_degree[t3] += 1

    # Initialize queue with zero in-degree nodes
    for i in range(n):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    # Process queue
    while head < tail:
        u = queue[head]
        head += 1
        order[idx] = u
        idx += 1

        for j in range(adj_count[u]):
            v = adj[u * max_edges_per_node + j]
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue[tail] = v
                tail += 1

    mid = idx // 2
    cdef int order_first = order[0]
    cdef int order_last = order[idx - 1]
    cdef int order_mid = order[mid]

    free(adj)
    free(adj_count)
    free(in_degree)
    free(queue)
    free(order)

    return (order_first, order_last, order_mid)
