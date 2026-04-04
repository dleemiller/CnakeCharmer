# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""BFS shortest-path distances from node 0 in a sparse deterministic graph (Cython-optimized).

Keywords: graph, bfs, shortest path, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def bfs_shortest_paths(int n):
    """BFS from node 0 in a deterministic graph built with edges (i,(i+1)%n) and (i,(i*3+7)%n).

    Uses CSR adjacency representation fully in C arrays.
    """
    # Each node has exactly 2 outgoing edges -> 2*n total edges
    cdef int num_edges = 2 * n
    cdef int *degree = <int *>malloc(n * sizeof(int))
    cdef int *adj_start = <int *>malloc((n + 1) * sizeof(int))
    cdef int *adj = <int *>malloc(num_edges * sizeof(int))
    cdef int *dist = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if not degree or not adj_start or not adj or not dist or not queue:
        if degree: free(degree)
        if adj_start: free(adj_start)
        if adj: free(adj)
        if dist: free(dist)
        if queue: free(queue)
        raise MemoryError()

    cdef int i, u, v, head, tail
    cdef long long total
    cdef int max_dist, num_reachable

    with nogil:
        # Each node has exactly 2 out-edges
        for i in range(n):
            degree[i] = 2

        # Build CSR adj_start
        adj_start[0] = 0
        for i in range(n):
            adj_start[i + 1] = adj_start[i] + degree[i]

        # Fill adj array (reuse degree as offset counter)
        for i in range(n):
            degree[i] = 0

        for i in range(n):
            # edge (i, (i+1)%n)
            adj[adj_start[i] + degree[i]] = (i + 1) % n
            degree[i] += 1
            # edge (i, (i*3+7)%n)
            adj[adj_start[i] + degree[i]] = (i * 3 + 7) % n
            degree[i] += 1

        # Initialize dist
        for i in range(n):
            dist[i] = -1

        # BFS from node 0
        dist[0] = 0
        queue[0] = 0
        head = 0
        tail = 1

        while head < tail:
            u = queue[head]
            head += 1
            for i in range(adj_start[u], adj_start[u + 1]):
                v = adj[i]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue[tail] = v
                    tail += 1

        # Aggregate results
        total = 0
        max_dist = 0
        num_reachable = 0
        for i in range(n):
            if dist[i] != -1:
                total += dist[i]
                if dist[i] > max_dist:
                    max_dist = dist[i]
                num_reachable += 1

    free(degree)
    free(adj_start)
    free(adj)
    free(dist)
    free(queue)

    return (total % (10 ** 9), max_dist, num_reachable)
