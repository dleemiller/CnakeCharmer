# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
BFS shortest path distances from node 0 on a deterministic graph (Cython-optimized).

Keywords: graph, bfs, shortest path, breadth-first search, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def bfs_shortest_path(int n):
    """Compute sum of shortest path distances from node 0 using C arrays for BFS.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Sum of shortest path distances from node 0 to all reachable nodes.
    """
    cdef int EDGES_PER_NODE = 3
    cdef int i, u, v, head, tail

    # Flat adjacency list: node i has edges at adj[i*3], adj[i*3+1], adj[i*3+2]
    cdef int *adj = <int *>malloc(n * EDGES_PER_NODE * sizeof(int))
    cdef int *dist = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if adj == NULL or dist == NULL or queue == NULL:
        if adj != NULL:
            free(adj)
        if dist != NULL:
            free(dist)
        if queue != NULL:
            free(queue)
        raise MemoryError("Failed to allocate BFS arrays")

    # Build adjacency list
    for i in range(n):
        adj[i * 3] = (i * 3 + 1) % n
        adj[i * 3 + 1] = (i * 7 + 2) % n
        adj[i * 3 + 2] = (i * 11 + 3) % n

    # Initialize distances to -1
    memset(dist, -1, n * sizeof(int))
    dist[0] = 0
    queue[0] = 0
    head = 0
    tail = 1

    # BFS
    while head < tail:
        u = queue[head]
        head += 1
        for i in range(EDGES_PER_NODE):
            v = adj[u * 3 + i]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue[tail] = v
                tail += 1

    # Sum distances
    cdef long long total = 0
    for i in range(n):
        if dist[i] != -1:
            total += dist[i]

    free(adj)
    free(dist)
    free(queue)
    return total
