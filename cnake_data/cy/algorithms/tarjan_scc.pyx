# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Tarjan's strongly connected components algorithm on a deterministic graph.

Keywords: algorithms, graph, tarjan, scc, strongly connected, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300000,))
def tarjan_scc(int n):
    """Find SCCs in a deterministic directed graph with n nodes."""
    # Adjacency: each node has 1-2 edges
    cdef int *edge1 = <int *>malloc(n * sizeof(int))
    cdef int *edge2 = <int *>malloc(n * sizeof(int))
    cdef int *num_edges = <int *>malloc(n * sizeof(int))
    cdef int *indices = <int *>malloc(n * sizeof(int))
    cdef int *lowlinks = <int *>malloc(n * sizeof(int))
    cdef char *on_stack = <char *>malloc(n * sizeof(char))
    cdef int *stack = <int *>malloc(n * sizeof(int))
    # SCC sizes
    cdef int *scc_sizes = <int *>malloc(n * sizeof(int))
    # Iterative call stack: (node, neighbor_index)
    cdef int *cs_node = <int *>malloc(n * sizeof(int))
    cdef int *cs_ni = <int *>malloc(n * sizeof(int))

    if not edge1 or not edge2 or not num_edges or not indices or not lowlinks or not on_stack or not stack or not scc_sizes or not cs_node or not cs_ni:
        if edge1: free(edge1)
        if edge2: free(edge2)
        if num_edges: free(num_edges)
        if indices: free(indices)
        if lowlinks: free(lowlinks)
        if on_stack: free(on_stack)
        if stack: free(stack)
        if scc_sizes: free(scc_sizes)
        if cs_node: free(cs_node)
        if cs_ni: free(cs_ni)
        raise MemoryError()

    cdef int i, v, w, node, ni, parent
    cdef int index_counter = 0
    cdef int stack_top = 0
    cdef int cs_top = 0
    cdef int num_scc = 0
    cdef int scc_count
    cdef int e1, e2

    # Build adjacency
    for i in range(n):
        e1 = (i * 3 + 7) % n
        e2 = (i * 5 + 11) % n
        edge1[i] = e1
        if e1 == e2:
            edge2[i] = -1
            num_edges[i] = 1
        else:
            edge2[i] = e2
            num_edges[i] = 2

    for i in range(n):
        indices[i] = -1
        lowlinks[i] = -1
        on_stack[i] = 0

    for v in range(n):
        if indices[v] != -1:
            continue

        # Start strongconnect(v)
        cs_top = 0
        cs_node[0] = v
        cs_ni[0] = 0
        indices[v] = index_counter
        lowlinks[v] = index_counter
        index_counter += 1
        stack[stack_top] = v
        stack_top += 1
        on_stack[v] = 1

        while cs_top >= 0:
            node = cs_node[cs_top]
            ni = cs_ni[cs_top]

            if ni < num_edges[node]:
                cs_ni[cs_top] = ni + 1
                if ni == 0:
                    w = edge1[node]
                else:
                    w = edge2[node]

                if indices[w] == -1:
                    indices[w] = index_counter
                    lowlinks[w] = index_counter
                    index_counter += 1
                    stack[stack_top] = w
                    stack_top += 1
                    on_stack[w] = 1
                    cs_top += 1
                    cs_node[cs_top] = w
                    cs_ni[cs_top] = 0
                elif on_stack[w]:
                    if lowlinks[node] > indices[w]:
                        lowlinks[node] = indices[w]
            else:
                if lowlinks[node] == indices[node]:
                    scc_count = 0
                    while True:
                        stack_top -= 1
                        w = stack[stack_top]
                        on_stack[w] = 0
                        scc_count += 1
                        if w == node:
                            break
                    scc_sizes[num_scc] = scc_count
                    num_scc += 1

                cs_top -= 1
                if cs_top >= 0:
                    parent = cs_node[cs_top]
                    if lowlinks[parent] > lowlinks[node]:
                        lowlinks[parent] = lowlinks[node]

    cdef int largest = 0
    cdef int smallest = n + 1
    for i in range(num_scc):
        if scc_sizes[i] > largest:
            largest = scc_sizes[i]
        if scc_sizes[i] < smallest:
            smallest = scc_sizes[i]

    cdef int result_num = num_scc
    cdef int result_largest = largest
    cdef int result_smallest = smallest

    free(edge1)
    free(edge2)
    free(num_edges)
    free(indices)
    free(lowlinks)
    free(on_stack)
    free(stack)
    free(scc_sizes)
    free(cs_node)
    free(cs_ni)

    return (result_num, result_largest, result_smallest)
