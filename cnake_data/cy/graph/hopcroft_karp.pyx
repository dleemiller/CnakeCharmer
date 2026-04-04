# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Hopcroft-Karp maximum bipartite matching on a deterministic graph (Cython-optimized).

Keywords: graph, bipartite, matching, hopcroft-karp, bfs, dfs, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

cdef int DEGREE = 5
cdef int[5][2] PAIRS


cdef int _dfs(int u, int *dist, int *match_l, int *match_r,
               int *adj, int degree, int n, int INF) noexcept nogil:
    """DFS augmentation along the BFS layered graph."""
    cdef int i, v, w
    for i in range(degree):
        v = adj[u * degree + i]
        w = match_r[v]
        if w == -1 or (dist[w] == dist[u] + 1 and
                        _dfs(w, dist, match_l, match_r, adj, degree, n, INF)):
            match_l[u] = v
            match_r[v] = u
            return 1
    dist[u] = INF
    return 0


@cython_benchmark(syntax="cy", args=(2000,))
def hopcroft_karp(int n):
    """Maximum bipartite matching using Hopcroft-Karp with C arrays.

    Returns:
        Tuple of (matching_size, matched_right_checksum).
    """
    cdef int INF = n + 1
    cdef int degree = 5
    cdef int ps[5]
    cdef int qs[5]
    ps[0] = 3;  qs[0] = 1
    ps[1] = 7;  qs[1] = 2
    ps[2] = 11; qs[2] = 3
    ps[3] = 13; qs[3] = 5
    ps[4] = 17; qs[4] = 7

    cdef int *adj    = <int *>malloc(n * degree * sizeof(int))
    cdef int *match_l = <int *>malloc(n * sizeof(int))
    cdef int *match_r = <int *>malloc(n * sizeof(int))
    cdef int *dist   = <int *>malloc(n * sizeof(int))
    cdef int *queue  = <int *>malloc(n * sizeof(int))

    if adj == NULL or match_l == NULL or match_r == NULL or dist == NULL or queue == NULL:
        if adj != NULL: free(adj)
        if match_l != NULL: free(match_l)
        if match_r != NULL: free(match_r)
        if dist != NULL: free(dist)
        if queue != NULL: free(queue)
        raise MemoryError()

    cdef int u, v, w, i, head, tail, matching
    cdef bint found
    cdef long long checksum

    # Build flat adjacency array
    for u in range(n):
        for i in range(degree):
            adj[u * degree + i] = (u * ps[i] + qs[i]) % n

    # Initialise matching to "unmatched"
    for u in range(n):
        match_l[u] = -1
        match_r[u] = -1

    matching = 0

    while True:
        # BFS to build layered graph
        head = 0; tail = 0
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                queue[tail] = u
                tail += 1
            else:
                dist[u] = INF
        found = False
        while head < tail:
            u = queue[head]; head += 1
            for i in range(degree):
                v = adj[u * degree + i]
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue[tail] = w
                    tail += 1
        if not found:
            break

        # DFS augmentation
        for u in range(n):
            if match_l[u] == -1:
                if _dfs(u, dist, match_l, match_r, adj, degree, n, INF):
                    matching += 1

    checksum = 0
    for u in range(n):
        if match_l[u] != -1:
            checksum += match_l[u]

    free(adj); free(match_l); free(match_r); free(dist); free(queue)
    return (matching, checksum)
