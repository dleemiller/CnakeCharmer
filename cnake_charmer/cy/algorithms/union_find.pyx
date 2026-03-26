# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Disjoint set union-find with path compression and union by rank.

Keywords: algorithms, union find, disjoint set, path compression, connected components, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def union_find(int n):
    """Perform union-find operations on n elements with deterministic edges."""
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *rank = <int *>malloc(n * sizeof(int))
    cdef int *comp_size = <int *>malloc(n * sizeof(int))

    if not parent or not rank or not comp_size:
        if parent: free(parent)
        if rank: free(rank)
        if comp_size: free(comp_size)
        raise MemoryError()

    cdef int i, a, b, ra, rb, tmp
    cdef int unions_done = 0
    cdef int num_edges = 3 * n / 2
    cdef int num_components = 0
    cdef int largest = 0

    # Initialize
    for i in range(n):
        parent[i] = i
        rank[i] = 0

    # Inline find with path halving
    for i in range(num_edges):
        a = (i * 31 + 7) % n
        b = (i * 67 + 13) % n
        if a == b:
            continue

        # Find root of a
        ra = a
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]

        # Find root of b
        rb = b
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]

        if ra == rb:
            continue

        # Union by rank
        if rank[ra] < rank[rb]:
            tmp = ra
            ra = rb
            rb = tmp
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        unions_done += 1

    # Count components and find largest
    memset(comp_size, 0, n * sizeof(int))
    for i in range(n):
        # Find root
        ra = i
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        comp_size[ra] += 1

    for i in range(n):
        if comp_size[i] > 0:
            num_components += 1
            if comp_size[i] > largest:
                largest = comp_size[i]

    free(parent)
    free(rank)
    free(comp_size)
    return (num_components, largest, unions_done)
