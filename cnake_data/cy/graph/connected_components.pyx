# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count connected components using union-find on a deterministic graph (Cython-optimized).

Keywords: graph, union-find, connected components, disjoint set, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def connected_components(int n):
    """Count connected components using C-array union-find with path compression.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (number of connected components, size of largest component).
    """
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *rank = <int *>malloc(n * sizeof(int))
    cdef int *comp_size = <int *>malloc(n * sizeof(int))

    if parent == NULL or rank == NULL or comp_size == NULL:
        if parent != NULL:
            free(parent)
        if rank != NULL:
            free(rank)
        if comp_size != NULL:
            free(comp_size)
        raise MemoryError("Failed to allocate union-find arrays")

    cdef int i, u, v, ru, rv, m, tmp

    # Initialize
    for i in range(n):
        parent[i] = i
        rank[i] = 0
        comp_size[i] = 0

    # Process edges
    m = n * 2
    for i in range(m):
        u = (i * 7 + 3) % n
        v = (i * 13 + 7) % n

        # Find root of u with path halving
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]

        # Find root of v with path halving
        rv = v
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]

        # Union by rank
        if ru != rv:
            if rank[ru] < rank[rv]:
                tmp = ru
                ru = rv
                rv = tmp
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1

    # Count components and find largest
    cdef int count = 0
    cdef int largest_size = 0
    for i in range(n):
        # Find root of i
        ru = i
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        comp_size[ru] += 1
        if ru == i:
            count += 1

    for i in range(n):
        if comp_size[i] > largest_size:
            largest_size = comp_size[i]

    free(parent)
    free(rank)
    free(comp_size)
    return (count, largest_size)
