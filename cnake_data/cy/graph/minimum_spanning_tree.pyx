# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Kruskal's minimum spanning tree (Cython-optimized).

Keywords: graph, MST, Kruskal, union-find, qsort, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_data.benchmarks import cython_benchmark


cdef struct Edge:
    int weight
    int u
    int v


cdef int _compare_edges(const void *a, const void *b) noexcept nogil:
    cdef int wa = (<Edge *>a).weight
    cdef int wb = (<Edge *>b).weight
    if wa < wb:
        return -1
    elif wa > wb:
        return 1
    return 0


cdef int _find(int *parent, int x) noexcept nogil:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@cython_benchmark(syntax="cy", args=(50000,))
def minimum_spanning_tree(int n):
    """Compute MST weight using Kruskal's algorithm on n nodes.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (total MST weight, edge count, max edge weight in MST).
    """
    cdef int m = 3 * n
    cdef int i, j, ru, rv, edge_count
    cdef long long total_weight
    cdef int max_edge_weight

    cdef Edge *edges = <Edge *>malloc(m * sizeof(Edge))
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *rank_arr = <int *>malloc(n * sizeof(int))

    if not edges or not parent or not rank_arr:
        free(edges)
        free(parent)
        free(rank_arr)
        raise MemoryError()

    # Build edges
    cdef int idx = 0
    for i in range(n):
        for j in range(1, 4):
            edges[idx].weight = (i * j + 3) % 100
            edges[idx].u = i
            edges[idx].v = (i * 7 + j) % n
            idx += 1

    # Sort edges by weight
    qsort(edges, m, sizeof(Edge), _compare_edges)

    # Initialize union-find
    for i in range(n):
        parent[i] = i
        rank_arr[i] = 0

    total_weight = 0
    edge_count = 0
    max_edge_weight = 0

    for i in range(m):
        ru = _find(parent, edges[i].u)
        rv = _find(parent, edges[i].v)
        if ru != rv:
            if rank_arr[ru] < rank_arr[rv]:
                parent[ru] = rv
            elif rank_arr[ru] > rank_arr[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank_arr[ru] += 1
            total_weight += edges[i].weight
            edge_count += 1
            if edges[i].weight > max_edge_weight:
                max_edge_weight = edges[i].weight
            if edge_count == n - 1:
                break

    free(edges)
    free(parent)
    free(rank_arr)
    return (total_weight, edge_count, max_edge_weight)
