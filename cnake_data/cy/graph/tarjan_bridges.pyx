# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find bridges in an undirected graph using Tarjan's algorithm (Cython-optimized).

Keywords: graph, bridges, tarjan, dfs, cut edges, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def tarjan_bridges(int n):
    """Find bridges in a deterministic undirected graph using C arrays."""
    if n < 2:
        return (0, 0)

    cdef int *degree = <int *>malloc(n * sizeof(int))
    cdef int *disc = <int *>malloc(n * sizeof(int))
    cdef int *low = <int *>malloc(n * sizeof(int))
    cdef int *stack_node = <int *>malloc(n * sizeof(int))
    cdef int *stack_parent = <int *>malloc(n * sizeof(int))
    cdef int *stack_idx = <int *>malloc(n * sizeof(int))

    if not degree or not disc or not low or not stack_node or not stack_parent or not stack_idx:
        if degree: free(degree)
        if disc: free(disc)
        if low: free(low)
        if stack_node: free(stack_node)
        if stack_parent: free(stack_parent)
        if stack_idx: free(stack_idx)
        raise MemoryError()

    cdef int i, k, parent, total_edges = 0

    memset(degree, 0, n * sizeof(int))

    # Count edges: binary tree + cross-edges every 6th node
    for i in range(1, n):
        parent = (i - 1) / 2
        degree[i] += 1
        degree[parent] += 1
        total_edges += 1
    for i in range(0, n, 6):
        k = (i * 7 + 3) % n
        if k != i:
            degree[i] += 1
            degree[k] += 1
            total_edges += 1

    # Build CSR adjacency
    cdef int *adj_offset = <int *>malloc((n + 1) * sizeof(int))
    cdef int *adj_list = <int *>malloc(2 * total_edges * sizeof(int))
    cdef int *adj_pos = <int *>malloc(n * sizeof(int))

    if not adj_offset or not adj_list or not adj_pos:
        free(degree); free(disc); free(low)
        free(stack_node); free(stack_parent); free(stack_idx)
        if adj_offset: free(adj_offset)
        if adj_list: free(adj_list)
        if adj_pos: free(adj_pos)
        raise MemoryError()

    adj_offset[0] = 0
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + degree[i]
    for i in range(n):
        adj_pos[i] = adj_offset[i]

    # Fill adjacency: binary tree
    for i in range(1, n):
        parent = (i - 1) / 2
        adj_list[adj_pos[i]] = parent
        adj_pos[i] += 1
        adj_list[adj_pos[parent]] = i
        adj_pos[parent] += 1
    # Cross-edges
    for i in range(0, n, 6):
        k = (i * 7 + 3) % n
        if k != i:
            adj_list[adj_pos[i]] = k
            adj_pos[i] += 1
            adj_list[adj_pos[k]] = i
            adj_pos[k] += 1

    # Initialize
    memset(disc, -1, n * sizeof(int))
    memset(low, -1, n * sizeof(int))

    cdef int timer = 0
    cdef int stack_top
    cdef int node, parent_node, neighbor, idx, start
    cdef int bridge_count = 0
    cdef long bridge_min_sum = 0

    # Iterative DFS
    for start in range(n):
        if disc[start] != -1:
            continue

        stack_top = 0
        stack_node[0] = start
        stack_parent[0] = -1
        stack_idx[0] = adj_offset[start]
        disc[start] = timer
        low[start] = timer
        timer += 1

        while stack_top >= 0:
            node = stack_node[stack_top]
            idx = stack_idx[stack_top]

            if idx < adj_offset[node + 1]:
                stack_idx[stack_top] = idx + 1
                neighbor = adj_list[idx]
                if disc[neighbor] == -1:
                    disc[neighbor] = timer
                    low[neighbor] = timer
                    timer += 1
                    stack_top += 1
                    stack_node[stack_top] = neighbor
                    stack_parent[stack_top] = node
                    stack_idx[stack_top] = adj_offset[neighbor]
                elif neighbor != stack_parent[stack_top]:
                    if disc[neighbor] < low[node]:
                        low[node] = disc[neighbor]
            else:
                stack_top -= 1
                if stack_top >= 0:
                    parent_node = stack_node[stack_top]
                    if low[node] < low[parent_node]:
                        low[parent_node] = low[node]
                    # Bridge condition
                    if low[node] > disc[parent_node]:
                        bridge_count += 1
                        if node < parent_node:
                            bridge_min_sum += node
                        else:
                            bridge_min_sum += parent_node

    free(degree)
    free(disc)
    free(low)
    free(stack_node)
    free(stack_parent)
    free(stack_idx)
    free(adj_offset)
    free(adj_list)
    free(adj_pos)

    return (bridge_count, bridge_min_sum)
