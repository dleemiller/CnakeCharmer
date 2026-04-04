# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Kruskal's MST using union-find on a deterministic weighted edge set (Cython-optimized).

Keywords: graph, MST, Kruskal, union-find, spanning tree, cython, benchmark
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


@cython_benchmark(syntax="cy", args=(80000,))
def kruskal_mst(int n):
    """Compute MST using Kruskal's algorithm with union-find and C arrays."""
    cdef int m = 2 * n
    cdef int i, ru, rv, edge_count
    cdef long long total_weight
    cdef int min_edge_weight

    cdef Edge *edges = <Edge *>malloc(m * sizeof(Edge))
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *rank_arr = <int *>malloc(n * sizeof(int))

    if not edges or not parent or not rank_arr:
        if edges: free(edges)
        if parent: free(parent)
        if rank_arr: free(rank_arr)
        raise MemoryError()

    # Build edges
    for i in range(n):
        edges[2 * i].weight = ((i * 3 + 2) % 97) + 1
        edges[2 * i].u = i
        edges[2 * i].v = (i * 5 + 1) % n
        edges[2 * i + 1].weight = ((i * 7 + 5) % 89) + 1
        edges[2 * i + 1].u = i
        edges[2 * i + 1].v = (i * 11 + 7) % n

    # Sort edges by weight
    qsort(edges, m, sizeof(Edge), _compare_edges)

    # Initialize union-find
    for i in range(n):
        parent[i] = i
        rank_arr[i] = 0

    total_weight = 0
    edge_count = 0
    min_edge_weight = 0

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
            if edge_count == 1:
                min_edge_weight = edges[i].weight
            if edge_count == n - 1:
                break

    free(edges)
    free(parent)
    free(rank_arr)
    return (total_weight, edge_count, min_edge_weight)
