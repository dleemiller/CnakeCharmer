# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""A* pathfinding on an n x n grid with deterministic obstacles.

Keywords: algorithms, pathfinding, a-star, grid, search, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def a_star_grid(int n):
    """Find shortest path on n x n grid from top-left to bottom-right."""
    cdef int nn = n * n
    cdef char *blocked = <char *>malloc(nn * sizeof(char))
    cdef int *g_score = <int *>malloc(nn * sizeof(int))
    cdef int *came_from = <int *>malloc(nn * sizeof(int))
    cdef char *in_closed = <char *>malloc(nn * sizeof(char))
    # Heap: store (f, node) as pairs in flat array
    cdef int heap_cap = nn * 2
    cdef int *heap_f = <int *>malloc(heap_cap * sizeof(int))
    cdef int *heap_n = <int *>malloc(heap_cap * sizeof(int))
    cdef int *path_buf = <int *>malloc(nn * sizeof(int))

    if not blocked or not g_score or not came_from or not in_closed or not heap_f or not heap_n or not path_buf:
        if blocked: free(blocked)
        if g_score: free(g_score)
        if came_from: free(came_from)
        if in_closed: free(in_closed)
        if heap_f: free(heap_f)
        if heap_n: free(heap_n)
        if path_buf: free(path_buf)
        raise MemoryError()

    cdef int i, j, d, ci, cj, ni, nj, neighbor, current, current_g, tentative_g
    cdef int h, goal, nodes_explored, min_idx, heap_size
    cdef int path_length, mid_node, path_midpoint_x, pos, idx
    cdef int INF = nn + 1
    cdef int dx[4]
    cdef int dy[4]
    dx[0] = -1; dx[1] = 1; dx[2] = 0; dx[3] = 0
    dy[0] = 0; dy[1] = 0; dy[2] = -1; dy[3] = 1

    # Build obstacle map
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13) % 10 == 0:
                blocked[i * n + j] = 1
            else:
                blocked[i * n + j] = 0
    blocked[0] = 0
    blocked[(n - 1) * n + (n - 1)] = 0

    for i in range(nn):
        g_score[i] = INF
        came_from[i] = -1
        in_closed[i] = 0
    g_score[0] = 0

    goal = (n - 1) * n + (n - 1)
    nodes_explored = 0
    heap_size = 1
    heap_f[0] = (n - 1) + (n - 1)
    heap_n[0] = 0

    while heap_size > 0:
        # Find min in heap
        min_idx = 0
        for i in range(1, heap_size):
            if heap_f[i] < heap_f[min_idx]:
                min_idx = i
        current = heap_n[min_idx]
        heap_size -= 1
        heap_f[min_idx] = heap_f[heap_size]
        heap_n[min_idx] = heap_n[heap_size]

        if in_closed[current]:
            continue

        in_closed[current] = 1
        nodes_explored += 1

        if current == goal:
            break

        ci = current / n
        cj = current % n
        current_g = g_score[current]

        for d in range(4):
            ni = ci + dx[d]
            nj = cj + dy[d]
            if 0 <= ni < n and 0 <= nj < n:
                neighbor = ni * n + nj
                if blocked[neighbor] == 0 and in_closed[neighbor] == 0:
                    tentative_g = current_g + 1
                    if tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        came_from[neighbor] = current
                        h = abs(ni - (n - 1)) + abs(nj - (n - 1))
                        # Add to heap (grow if needed not likely)
                        heap_f[heap_size] = tentative_g + h
                        heap_n[heap_size] = neighbor
                        heap_size += 1

    if g_score[goal] == INF:
        free(blocked)
        free(g_score)
        free(came_from)
        free(in_closed)
        free(heap_f)
        free(heap_n)
        free(path_buf)
        return (-1, nodes_explored, -1)

    path_length = g_score[goal]

    # Reconstruct path
    pos = goal
    idx = path_length
    while pos != -1:
        path_buf[idx] = pos
        pos = came_from[pos]
        idx -= 1

    mid_node = path_buf[path_length / 2]
    path_midpoint_x = mid_node / n

    free(blocked)
    free(g_score)
    free(came_from)
    free(in_closed)
    free(heap_f)
    free(heap_n)
    free(path_buf)

    return (path_length, nodes_explored, path_midpoint_x)
