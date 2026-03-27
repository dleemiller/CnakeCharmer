# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count strongly connected components using Tarjan's algorithm (Cython-optimized).

Keywords: graph, strongly connected components, Tarjan, SCC, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def strongly_connected(int n):
    """Count strongly connected components using Tarjan's algorithm."""
    cdef int i, v, w, ei, parent, scc_size
    cdef int idx_counter = 0
    cdef int scc_count = 0
    cdef int largest_scc_size = 0
    cdef int stack_top = -1
    cdef int work_top = -1

    # Adjacency: each node has exactly 2 edges
    cdef int *adj0 = <int *>malloc(n * sizeof(int))
    cdef int *adj1 = <int *>malloc(n * sizeof(int))
    cdef int *index_arr = <int *>malloc(n * sizeof(int))
    cdef int *lowlink = <int *>malloc(n * sizeof(int))
    cdef int *on_stack = <int *>malloc(n * sizeof(int))
    cdef int *stack = <int *>malloc(n * sizeof(int))
    # Work stack: pairs (node, edge_index) stored as 2 arrays
    cdef int *work_v = <int *>malloc(n * sizeof(int))
    cdef int *work_ei = <int *>malloc(n * sizeof(int))

    if not adj0 or not adj1 or not index_arr or not lowlink or not on_stack or not stack or not work_v or not work_ei:
        free(adj0); free(adj1); free(index_arr); free(lowlink)
        free(on_stack); free(stack); free(work_v); free(work_ei)
        raise MemoryError()

    for i in range(n):
        adj0[i] = (i * 3 + 1) % n
        adj1[i] = (i * 7 + 2) % n
        index_arr[i] = -1
        lowlink[i] = -1
        on_stack[i] = 0

    for i in range(n):
        if index_arr[i] != -1:
            continue

        # Push starting node
        work_top += 1
        work_v[work_top] = i
        work_ei[work_top] = 0

        while work_top >= 0:
            v = work_v[work_top]
            ei = work_ei[work_top]

            if ei == 0:
                index_arr[v] = idx_counter
                lowlink[v] = idx_counter
                idx_counter += 1
                stack_top += 1
                stack[stack_top] = v
                on_stack[v] = 1

            # Try edges
            if ei == 0:
                w = adj0[v]
                if index_arr[w] == -1:
                    work_ei[work_top] = 1
                    work_top += 1
                    work_v[work_top] = w
                    work_ei[work_top] = 0
                    continue
                elif on_stack[w]:
                    if lowlink[v] > lowlink[w]:
                        lowlink[v] = lowlink[w]
                ei = 1

            if ei == 1:
                w = adj1[v]
                if index_arr[w] == -1:
                    work_ei[work_top] = 2
                    work_top += 1
                    work_v[work_top] = w
                    work_ei[work_top] = 0
                    continue
                elif on_stack[w]:
                    if lowlink[v] > lowlink[w]:
                        lowlink[v] = lowlink[w]

            # Done with v
            work_top -= 1
            if work_top >= 0:
                parent = work_v[work_top]
                if lowlink[parent] > lowlink[v]:
                    lowlink[parent] = lowlink[v]

            if lowlink[v] == index_arr[v]:
                scc_count += 1
                scc_size = 0
                while True:
                    w = stack[stack_top]
                    stack_top -= 1
                    on_stack[w] = 0
                    scc_size += 1
                    if w == v:
                        break
                if scc_size > largest_scc_size:
                    largest_scc_size = scc_size

    free(adj0); free(adj1); free(index_arr); free(lowlink)
    free(on_stack); free(stack); free(work_v); free(work_ei)
    return (scc_count, largest_scc_size)
