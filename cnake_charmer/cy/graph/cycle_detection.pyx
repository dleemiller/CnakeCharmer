# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count back edges (cycles) via DFS in a directed graph (Cython-optimized).

Keywords: graph, cycle, dfs, detection, back edge, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def cycle_detection(int n):
    """Count back edges found via iterative DFS using C arrays.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (number of back edges, source node of last back edge or -1).
    """
    cdef int i, start, u, v, ei
    cdef int back_edges = 0
    cdef int last_back_src = -1
    cdef int WHITE = 0, GRAY = 1, BLACK = 2

    cdef int *adj_0 = <int *>malloc(n * sizeof(int))
    cdef int *adj_1 = <int *>malloc(n * sizeof(int))
    cdef char *color = <char *>malloc(n * sizeof(char))
    # Stack: pairs of (node, edge_index) stored as two parallel arrays
    cdef int *stack_node = <int *>malloc(n * sizeof(int))
    cdef int *stack_ei = <int *>malloc(n * sizeof(int))

    if adj_0 == NULL or adj_1 == NULL or color == NULL or stack_node == NULL or stack_ei == NULL:
        if adj_0 != NULL: free(adj_0)
        if adj_1 != NULL: free(adj_1)
        if color != NULL: free(color)
        if stack_node != NULL: free(stack_node)
        if stack_ei != NULL: free(stack_ei)
        raise MemoryError("Failed to allocate DFS arrays")

    # Build adjacency
    for i in range(n):
        adj_0[i] = (i * 3 + 1) % n
        adj_1[i] = (i * 7 + 2) % n
        color[i] = WHITE

    cdef int top

    for start in range(n):
        if color[start] != WHITE:
            continue
        # Iterative DFS
        top = 0
        stack_node[0] = start
        stack_ei[0] = 0
        color[start] = GRAY

        while top >= 0:
            u = stack_node[top]
            ei = stack_ei[top]
            if ei < 2:
                stack_ei[top] = ei + 1
                if ei == 0:
                    v = adj_0[u]
                else:
                    v = adj_1[u]
                if color[v] == WHITE:
                    color[v] = GRAY
                    top += 1
                    stack_node[top] = v
                    stack_ei[top] = 0
                elif color[v] == GRAY:
                    back_edges += 1
                    last_back_src = u
            else:
                color[u] = BLACK
                top -= 1

    free(adj_0)
    free(adj_1)
    free(color)
    free(stack_node)
    free(stack_ei)
    return (back_edges, last_back_src)
