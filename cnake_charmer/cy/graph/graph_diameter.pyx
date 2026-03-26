# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find the diameter of a graph using repeated BFS (Cython-optimized).

Keywords: graph, diameter, bfs, longest shortest path, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(4000,))
def graph_diameter(int n):
    """Compute graph diameter via BFS from multiple sources using C arrays."""
    if n < 2:
        return (0, 0, 0)

    cdef int *degree = <int *>malloc(n * sizeof(int))
    cdef int *dist = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if not degree or not dist or not queue:
        if degree: free(degree)
        if dist: free(dist)
        if queue: free(queue)
        raise MemoryError()

    cdef int i, j, k, total_edges = 0

    memset(degree, 0, n * sizeof(int))

    # Count edges: ring + cross-edges
    for i in range(n):
        j = (i + 1) % n
        degree[i] += 1
        degree[j] += 1
        total_edges += 1
    for i in range(n):
        k = (i * 3 + 5) % n
        if k != i:
            degree[i] += 1
            degree[k] += 1
            total_edges += 1

    # Build CSR adjacency
    cdef int *adj_offset = <int *>malloc((n + 1) * sizeof(int))
    cdef int *adj_list = <int *>malloc(2 * total_edges * sizeof(int))
    cdef int *adj_pos = <int *>malloc(n * sizeof(int))

    if not adj_offset or not adj_list or not adj_pos:
        free(degree); free(dist); free(queue)
        if adj_offset: free(adj_offset)
        if adj_list: free(adj_list)
        if adj_pos: free(adj_pos)
        raise MemoryError()

    adj_offset[0] = 0
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + degree[i]
    for i in range(n):
        adj_pos[i] = adj_offset[i]

    # Fill adjacency: ring
    for i in range(n):
        j = (i + 1) % n
        adj_list[adj_pos[i]] = j
        adj_pos[i] += 1
        adj_list[adj_pos[j]] = i
        adj_pos[j] += 1
    # Cross-edges
    for i in range(n):
        k = (i * 3 + 5) % n
        if k != i:
            adj_list[adj_pos[i]] = k
            adj_pos[i] += 1
            adj_list[adj_pos[k]] = i
            adj_pos[k] += 1

    cdef int sources[4]
    sources[0] = 0
    sources[1] = n / 4
    sources[2] = n / 2
    sources[3] = (3 * n) / 4

    cdef int diameter = 0
    cdef int eccentricity_sum = 0
    cdef long total_edges_visited = 0
    cdef int s, src, u, v, qfront, qback, max_dist, edges_visited

    for s in range(4):
        src = sources[s]
        memset(dist, -1, n * sizeof(int))
        dist[src] = 0
        qfront = 0
        qback = 0
        queue[qback] = src
        qback += 1
        max_dist = 0
        edges_visited = 0

        while qfront < qback:
            u = queue[qfront]
            qfront += 1
            for idx in range(adj_offset[u], adj_offset[u + 1]):
                v = adj_list[idx]
                edges_visited += 1
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue[qback] = v
                    qback += 1

        eccentricity_sum += max_dist
        total_edges_visited += edges_visited
        if max_dist > diameter:
            diameter = max_dist

    free(degree)
    free(dist)
    free(queue)
    free(adj_offset)
    free(adj_list)
    free(adj_pos)

    return (diameter, eccentricity_sum, total_edges_visited)
