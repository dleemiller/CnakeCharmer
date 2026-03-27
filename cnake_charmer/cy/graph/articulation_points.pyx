# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count articulation points in a deterministic graph via DFS (Cython-optimized).

Keywords: graph, articulation points, cut vertices, dfs, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def articulation_points(int n):
    """Count articulation points using Tarjan's algorithm with C arrays."""
    if n < 2:
        return 0

    # Build adjacency list using C arrays
    # First pass: count degree of each node
    cdef int *degree = <int *>malloc(n * sizeof(int))
    cdef int *disc = <int *>malloc(n * sizeof(int))
    cdef int *low = <int *>malloc(n * sizeof(int))
    cdef int *is_ap = <int *>malloc(n * sizeof(int))
    cdef int *child_count = <int *>malloc(n * sizeof(int))
    # Stack for iterative DFS: node, parent, adj_index
    cdef int *stack_node = <int *>malloc(3 * n * sizeof(int))
    cdef int *stack_parent = <int *>malloc(3 * n * sizeof(int))
    cdef int *stack_idx = <int *>malloc(3 * n * sizeof(int))

    if not degree or not disc or not low or not is_ap or not child_count or not stack_node or not stack_parent or not stack_idx:
        if degree: free(degree)
        if disc: free(disc)
        if low: free(low)
        if is_ap: free(is_ap)
        if child_count: free(child_count)
        if stack_node: free(stack_node)
        if stack_parent: free(stack_parent)
        if stack_idx: free(stack_idx)
        raise MemoryError()

    cdef int i, k, parent, node, neighbor, idx
    cdef int timer = 0
    cdef int stack_top
    cdef int parent_node, start
    cdef int total_edges = 0
    cdef int count = 0
    cdef long ap_index_sum = 0

    memset(degree, 0, n * sizeof(int))

    # Count edges: tree edges + sparse cross-edges
    for i in range(1, n):
        parent = (i - 1) / 2
        degree[i] += 1
        degree[parent] += 1
        total_edges += 1
    for i in range(0, n, 5):
        k = (i * 3 + 7) % n
        if k != i:
            degree[i] += 1
            degree[k] += 1
            total_edges += 1

    # Build CSR-like adjacency
    cdef int *adj_offset = <int *>malloc((n + 1) * sizeof(int))
    cdef int *adj_list = <int *>malloc(2 * total_edges * sizeof(int))
    cdef int *adj_pos = <int *>malloc(n * sizeof(int))
    if not adj_offset or not adj_list or not adj_pos:
        free(degree)
        free(disc)
        free(low)
        free(is_ap)
        free(child_count)
        free(stack_node)
        free(stack_parent)
        free(stack_idx)
        if adj_offset: free(adj_offset)
        if adj_list: free(adj_list)
        if adj_pos: free(adj_pos)
        raise MemoryError()

    adj_offset[0] = 0
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + degree[i]
    for i in range(n):
        adj_pos[i] = adj_offset[i]

    # Fill adjacency
    for i in range(1, n):
        parent = (i - 1) / 2
        adj_list[adj_pos[i]] = parent
        adj_pos[i] += 1
        adj_list[adj_pos[parent]] = i
        adj_pos[parent] += 1
    for i in range(0, n, 5):
        k = (i * 3 + 7) % n
        if k != i:
            adj_list[adj_pos[i]] = k
            adj_pos[i] += 1
            adj_list[adj_pos[k]] = i
            adj_pos[k] += 1

    # Initialize
    memset(disc, -1, n * sizeof(int))
    memset(low, -1, n * sizeof(int))
    memset(is_ap, 0, n * sizeof(int))

    # Iterative DFS
    for start in range(n):
        if disc[start] != -1:
            continue

        memset(child_count, 0, n * sizeof(int))
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
                    child_count[node] += 1
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
                    if stack_parent[stack_top] == -1:
                        pass  # root handled after
                    else:
                        if low[node] >= disc[parent_node]:
                            is_ap[parent_node] = 1

        if child_count[start] > 1:
            is_ap[start] = 1

    for i in range(n):
        if is_ap[i]:
            count += 1
            ap_index_sum += i

    free(degree)
    free(disc)
    free(low)
    free(is_ap)
    free(child_count)
    free(stack_node)
    free(stack_parent)
    free(stack_idx)
    free(adj_offset)
    free(adj_list)
    free(adj_pos)
    return (count, ap_index_sum)


