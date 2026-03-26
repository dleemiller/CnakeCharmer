# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Max flow in a layered graph using Ford-Fulkerson with BFS (Cython-optimized).

Keywords: algorithms, max flow, Ford-Fulkerson, BFS, graph, network flow, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def max_flow(int n):
    """Compute max flow using Edmonds-Karp with C arrays for adjacency matrix."""
    if n < 2:
        return 0

    # Use flat adjacency matrix for capacity (n*n)
    # For n=1000 this is 4MB which is fine
    cdef int *cap = <int *>malloc(n * n * sizeof(int))
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if not cap or not parent or not queue:
        if cap: free(cap)
        if parent: free(parent)
        if queue: free(queue)
        raise MemoryError()

    cdef int i, j, j1, j2, u, v, head, tail
    cdef int total_flow, path_flow, c
    cdef int source = 0
    cdef int sink = n - 1
    cdef int n3 = n / 3

    # Initialize capacity matrix
    memset(cap, 0, n * n * sizeof(int))

    for i in range(n):
        j1 = i + 1
        if j1 < n:
            cap[i * n + j1] += i % 5 + 1
        j2 = (i + n3) % n
        if j2 != i:
            cap[i * n + j2] += i % 3 + 1

    total_flow = 0

    while True:
        # BFS to find augmenting path
        memset(parent, -1, n * sizeof(int))
        parent[source] = source
        queue[0] = source
        head = 0
        tail = 1
        while head < tail:
            u = queue[head]
            head += 1
            if u == sink:
                break
            for v in range(n):
                if cap[u * n + v] > 0 and parent[v] == -1:
                    parent[v] = u
                    queue[tail] = v
                    tail += 1

        if parent[sink] == -1:
            break

        # Find bottleneck
        path_flow = 2000000000  # large sentinel
        v = sink
        while v != source:
            u = parent[v]
            c = cap[u * n + v]
            if c < path_flow:
                path_flow = c
            v = u

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            cap[u * n + v] -= path_flow
            cap[v * n + u] += path_flow
            v = u

        total_flow += path_flow

    free(cap)
    free(parent)
    free(queue)
    return total_flow
