# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Check if a generated graph is bipartite using BFS coloring (Cython-optimized).

Keywords: bipartite, graph, bfs, coloring, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def bipartite_check(int n):
    """Check bipartiteness using BFS with C arrays for adjacency and queue."""
    # Each node has exactly 4 edges (2 outgoing become 4 with reverse)
    # But edges can overlap, so use a flat adjacency list with offsets
    # Max edges per node: we'll use linked lists via arrays

    cdef int *adj_target = <int *>malloc(4 * n * sizeof(int))
    cdef int *adj_next = <int *>malloc(4 * n * sizeof(int))
    cdef int *adj_head = <int *>malloc(n * sizeof(int))
    cdef int *color = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if not adj_target or not adj_next or not adj_head or not color or not queue:
        free(adj_target); free(adj_next); free(adj_head); free(color); free(queue)
        raise MemoryError()

    cdef int i, u, v, node, neighbor, start
    cdef int head, tail, edge_count, is_bipartite, colored_count, edge_idx

    memset(adj_head, -1, n * sizeof(int))
    edge_count = 0

    for i in range(n):
        u = (i * 3 + 1) % n
        v = (i * 7 + 2) % n

        # Add edge i -> u
        adj_target[edge_count] = u
        adj_next[edge_count] = adj_head[i]
        adj_head[i] = edge_count
        edge_count += 1

        # Add edge u -> i
        adj_target[edge_count] = i
        adj_next[edge_count] = adj_head[u]
        adj_head[u] = edge_count
        edge_count += 1

        # Add edge i -> v
        adj_target[edge_count] = v
        adj_next[edge_count] = adj_head[i]
        adj_head[i] = edge_count
        edge_count += 1

        # Add edge v -> i
        adj_target[edge_count] = i
        adj_next[edge_count] = adj_head[v]
        adj_head[v] = edge_count
        edge_count += 1

    memset(color, -1, n * sizeof(int))
    is_bipartite = 1
    colored_count = 0

    for start in range(n):
        if color[start] != -1:
            continue
        queue[0] = start
        color[start] = 0
        colored_count += 1
        head = 0
        tail = 1
        while head < tail:
            node = queue[head]
            head += 1
            edge_idx = adj_head[node]
            while edge_idx != -1:
                neighbor = adj_target[edge_idx]
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    colored_count += 1
                    queue[tail] = neighbor
                    tail += 1
                elif color[neighbor] == color[node]:
                    is_bipartite = 0
                edge_idx = adj_next[edge_idx]

    free(adj_target)
    free(adj_next)
    free(adj_head)
    free(color)
    free(queue)
    return (is_bipartite << 20) | colored_count
