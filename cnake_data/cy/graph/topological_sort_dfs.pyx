# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""DFS-based topological sort on a deterministic DAG (Cython-optimized).

Keywords: graph, topological sort, DFS, DAG, ordering, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def topological_sort_dfs(int n):
    """Compute sum of topological order positions using DFS-based topo sort with C arrays."""
    cdef int i, j, start, node, idx, nb
    cdef int order_count = 0

    # Build adjacency: each node has at most 1 edge
    cdef int *adj = <int *>malloc(n * sizeof(int))
    cdef char *has_edge = <char *>malloc(n * sizeof(char))
    cdef char *color = <char *>malloc(n * sizeof(char))  # 0=WHITE, 1=GRAY, 2=BLACK
    cdef int *order = <int *>malloc(n * sizeof(int))
    # Stack stores (node, idx) as pairs
    cdef int *stack_node = <int *>malloc(n * sizeof(int))
    cdef int *stack_idx = <int *>malloc(n * sizeof(int))
    cdef int stack_top = 0

    if not adj or not has_edge or not color or not order or not stack_node or not stack_idx:
        free(adj); free(has_edge); free(color); free(order)
        free(stack_node); free(stack_idx)
        raise MemoryError()

    memset(has_edge, 0, n * sizeof(char))
    memset(color, 0, n * sizeof(char))

    for i in range(n):
        j = (i * 3 + 1) % n
        if j > i:
            adj[i] = j
            has_edge[i] = 1

    # Iterative DFS
    for start in range(n):
        if color[start] != 0:
            continue
        stack_node[0] = start
        stack_idx[0] = 0
        stack_top = 1
        color[start] = 1

        while stack_top > 0:
            node = stack_node[stack_top - 1]
            idx = stack_idx[stack_top - 1]
            if has_edge[node] and idx < 1:
                stack_idx[stack_top - 1] = idx + 1
                nb = adj[node]
                if color[nb] == 0:
                    color[nb] = 1
                    stack_node[stack_top] = nb
                    stack_idx[stack_top] = 0
                    stack_top += 1
            else:
                color[node] = 2
                order[order_count] = node
                order_count += 1
                stack_top -= 1

    # Sum of positions (0 + 1 + ... + n-1)
    cdef long long total = 0
    for i in range(n):
        total += i

    free(adj)
    free(has_edge)
    free(color)
    free(order)
    free(stack_node)
    free(stack_idx)
    return total
