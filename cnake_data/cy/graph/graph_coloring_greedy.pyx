# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Greedy graph coloring on a deterministic sparse graph (Cython-optimized).

Keywords: graph coloring, greedy, chromatic, adjacency, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def graph_coloring_greedy(int n):
    """Greedy graph coloring and return number of colors used."""
    # Each node has at most 6 neighbors (3 forward + 3 reverse)
    # Use flat adjacency storage: adj_list[i*MAX_DEG .. i*MAX_DEG+deg[i]-1]
    cdef int MAX_DEG = 12
    cdef int *adj_list = <int *>malloc(n * MAX_DEG * sizeof(int))
    cdef int *deg = <int *>malloc(n * sizeof(int))
    cdef int *colors = <int *>malloc(n * sizeof(int))
    cdef unsigned char *used = <unsigned char *>malloc(256 * sizeof(unsigned char))

    if not adj_list or not deg or not colors or not used:
        if adj_list:
            free(adj_list)
        if deg:
            free(deg)
        if colors:
            free(colors)
        if used:
            free(used)
        raise MemoryError()

    cdef int i, j, target, color, max_color, neighbor, base
    cdef long color_checksum = 0
    cdef int num_colors

    memset(deg, 0, n * sizeof(int))

    # Build adjacency list
    for i in range(n):
        target = (i * 3 + 1) % n
        if target != i and deg[i] < MAX_DEG and deg[target] < MAX_DEG:
            adj_list[i * MAX_DEG + deg[i]] = target
            deg[i] += 1
            adj_list[target * MAX_DEG + deg[target]] = i
            deg[target] += 1
        target = (i * 7 + 2) % n
        if target != i and deg[i] < MAX_DEG and deg[target] < MAX_DEG:
            adj_list[i * MAX_DEG + deg[i]] = target
            deg[i] += 1
            adj_list[target * MAX_DEG + deg[target]] = i
            deg[target] += 1
        target = (i * 11 + 3) % n
        if target != i and deg[i] < MAX_DEG and deg[target] < MAX_DEG:
            adj_list[i * MAX_DEG + deg[i]] = target
            deg[i] += 1
            adj_list[target * MAX_DEG + deg[target]] = i
            deg[target] += 1

    # Color nodes
    for i in range(n):
        colors[i] = -1

    max_color = 0

    for i in range(n):
        memset(used, 0, 256 * sizeof(unsigned char))
        base = i * MAX_DEG
        for j in range(deg[i]):
            neighbor = adj_list[base + j]
            if colors[neighbor] >= 0 and colors[neighbor] < 256:
                used[colors[neighbor]] = 1

        color = 0
        while color < 256 and used[color]:
            color += 1
        colors[i] = color
        if color > max_color:
            max_color = color

    num_colors = max_color + 1
    color_checksum = 0
    for i in range(n):
        color_checksum += colors[i] * ((i * 31 + 7) % 1000)
    color_checksum = color_checksum % (10**9 + 7)

    free(adj_list)
    free(deg)
    free(colors)
    free(used)

    return (num_colors, color_checksum)
