# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count edges traversed in an Eulerian path on a directed graph (Cython-optimized).

Keywords: graph, euler, eulerian path, directed, traversal, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def euler_path(int n):
    """Count edges in Eulerian circuit using Hierholzer's with C arrays."""
    if n < 3:
        return 0

    cdef int EDGES_PER_NODE = 2
    cdef int total_edges = n * EDGES_PER_NODE
    cdef int i, v, u

    # Flat adjacency: node i has edges at adj[i*2] and adj[i*2+1]
    cdef int *adj = <int *>malloc(total_edges * sizeof(int))
    cdef int *edge_idx = <int *>malloc(n * sizeof(int))
    # Stack and path can hold at most total_edges + 1 entries
    cdef int *stack = <int *>malloc((total_edges + 1) * sizeof(int))
    cdef int stack_top = 0
    cdef int path_count = 0

    if not adj or not edge_idx or not stack:
        free(adj); free(edge_idx); free(stack)
        raise MemoryError()

    for i in range(n):
        adj[i * 2] = (i + 1) % n
        adj[i * 2 + 1] = (i + 2) % n
        edge_idx[i] = 0

    # Hierholzer's algorithm
    stack[0] = 0
    stack_top = 1

    while stack_top > 0:
        v = stack[stack_top - 1]
        if edge_idx[v] < EDGES_PER_NODE:
            u = adj[v * 2 + edge_idx[v]]
            edge_idx[v] += 1
            stack[stack_top] = u
            stack_top += 1
        else:
            stack_top -= 1
            path_count += 1

    free(adj)
    free(edge_idx)
    free(stack)
    return path_count - 1
