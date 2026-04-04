# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Max flow in a layered graph using Edmonds-Karp with edge-pair adjacency list (Cython).

Uses the standard competitive-programming edge-pair representation:
edges are stored in a flat array where edge i and i^1 are forward/reverse
pairs. Each node has a linked list of its edge indices. This gives O(E)
BFS instead of O(V) per node, and uses O(V+E) memory instead of O(V^2).

Keywords: algorithms, max flow, Ford-Fulkerson, BFS, graph, network flow, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20000,))
def max_flow(int n):
    """Compute max flow using Edmonds-Karp with edge-pair adjacency list."""
    if n < 2:
        return 0

    # Edge-pair representation: edge[i] and edge[i^1] are forward/reverse
    # Max edges: 2 forward per node × 2 (with reverse) = 4n
    cdef int max_edges = 4 * n + 4
    cdef int *edge_to = <int *>malloc(max_edges * sizeof(int))
    cdef int *edge_cap = <int *>malloc(max_edges * sizeof(int))
    cdef int *edge_next = <int *>malloc(max_edges * sizeof(int))
    cdef int *head = <int *>malloc(n * sizeof(int))
    cdef int *parent_edge = <int *>malloc(n * sizeof(int))
    cdef int *queue = <int *>malloc(n * sizeof(int))

    if not edge_to or not edge_cap or not edge_next or not head or not parent_edge or not queue:
        if edge_to: free(edge_to)
        if edge_cap: free(edge_cap)
        if edge_next: free(edge_next)
        if head: free(head)
        if parent_edge: free(parent_edge)
        if queue: free(queue)
        raise MemoryError()

    cdef int num_edges = 0
    cdef int i, u, v, e, qhead, qtail
    cdef int total_flow, path_flow
    cdef int source = 0
    cdef int sink = n - 1
    cdef int n3 = n / 3

    # Initialize adjacency list heads
    memset(head, -1, n * sizeof(int))

    # Add edge pair: forward (u->v, cap) and reverse (v->u, 0)
    for i in range(n):
        # Edge i -> i+1
        if i + 1 < n:
            # Forward edge
            e = num_edges
            edge_to[e] = i + 1
            edge_cap[e] = i % 5 + 1
            edge_next[e] = head[i]
            head[i] = e
            num_edges += 1
            # Reverse edge
            e = num_edges
            edge_to[e] = i
            edge_cap[e] = 0
            edge_next[e] = head[i + 1]
            head[i + 1] = e
            num_edges += 1

        # Edge i -> (i + n//3) % n
        v = (i + n3) % n
        if v != i:
            # Forward edge
            e = num_edges
            edge_to[e] = v
            edge_cap[e] = i % 3 + 1
            edge_next[e] = head[i]
            head[i] = e
            num_edges += 1
            # Reverse edge
            e = num_edges
            edge_to[e] = i
            edge_cap[e] = 0
            edge_next[e] = head[v]
            head[v] = e
            num_edges += 1

    total_flow = 0

    while True:
        # BFS to find augmenting path
        memset(parent_edge, -1, n * sizeof(int))
        parent_edge[source] = -2  # mark visited, no parent edge
        queue[0] = source
        qhead = 0
        qtail = 1
        while qhead < qtail:
            u = queue[qhead]
            qhead += 1
            if u == sink:
                break
            e = head[u]
            while e != -1:
                v = edge_to[e]
                if edge_cap[e] > 0 and parent_edge[v] == -1:
                    parent_edge[v] = e
                    queue[qtail] = v
                    qtail += 1
                e = edge_next[e]

        if parent_edge[sink] == -1:
            break

        # Find bottleneck
        path_flow = 2000000000
        v = sink
        while v != source:
            e = parent_edge[v]
            if edge_cap[e] < path_flow:
                path_flow = edge_cap[e]
            v = edge_to[e ^ 1]  # reverse edge points to parent

        # Update residual capacities
        v = sink
        while v != source:
            e = parent_edge[v]
            edge_cap[e] -= path_flow
            edge_cap[e ^ 1] += path_flow
            v = edge_to[e ^ 1]

        total_flow += path_flow

    free(edge_to)
    free(edge_cap)
    free(edge_next)
    free(head)
    free(parent_edge)
    free(queue)
    return total_flow
