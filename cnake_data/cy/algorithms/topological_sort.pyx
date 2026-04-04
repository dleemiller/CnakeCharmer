# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Topological sort of a DAG and reachability count from node 0 (Cython-optimized).

Keywords: algorithms, topological sort, DAG, graph, reachability, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def topological_sort(int n):
    """Topological sort a DAG, count reachable from node 0, using C arrays."""
    cdef int *adj_target = <int *>malloc(n * sizeof(int))
    cdef int *has_edge = <int *>malloc(n * sizeof(int))
    cdef int *in_degree = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))
    cdef int *topo_order = <int *>malloc(n * sizeof(int))
    cdef char *reachable = <char *>malloc(n * sizeof(char))

    if not adj_target or not has_edge or not in_degree or not queue or not topo_order or not reachable:
        if adj_target: free(adj_target)
        if has_edge: free(has_edge)
        if in_degree: free(in_degree)
        if queue: free(queue)
        if topo_order: free(topo_order)
        if reachable: free(reachable)
        raise MemoryError()

    cdef int i, target, node, neighbor, head, tail, topo_count, count

    # Each node has at most one outgoing edge: i -> (i*3+1)%n if target > i
    memset(in_degree, 0, n * sizeof(int))
    memset(has_edge, 0, n * sizeof(int))

    for i in range(n):
        target = (i * 3 + 1) % n
        if target > i:
            adj_target[i] = target
            has_edge[i] = 1
            in_degree[target] += 1
        else:
            adj_target[i] = -1

    # Kahn's algorithm
    head = 0
    tail = 0
    for i in range(n):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    topo_count = 0
    while head < tail:
        node = queue[head]
        head += 1
        topo_order[topo_count] = node
        topo_count += 1
        if has_edge[node]:
            neighbor = adj_target[node]
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue[tail] = neighbor
                tail += 1

    # Count reachable from node 0
    memset(reachable, 0, n * sizeof(char))
    reachable[0] = 1
    count = 0

    for i in range(topo_count):
        node = topo_order[i]
        if reachable[node]:
            count += 1
            if has_edge[node]:
                reachable[adj_target[node]] = 1

    free(adj_target)
    free(has_edge)
    free(in_degree)
    free(queue)
    free(topo_order)
    free(reachable)
    return count
