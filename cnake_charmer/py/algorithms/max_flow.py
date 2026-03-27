"""Max flow in a layered graph using Ford-Fulkerson with BFS.

Keywords: algorithms, max flow, Ford-Fulkerson, BFS, graph, network flow, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(20000,))
def max_flow(n: int) -> int:
    """Compute max flow in a layered graph with n nodes.

    Edges: i -> (i+1) with capacity (i%5+1),
           i -> (i + n//3) % n with capacity (i%3+1).
    Source = 0, Sink = n-1.
    Uses Ford-Fulkerson with BFS (Edmonds-Karp).

    Args:
        n: Number of nodes.

    Returns:
        Maximum flow value from source to sink.
    """
    if n < 2:
        return 0

    # Build adjacency list with capacities
    # capacity[i] is a dict: neighbor -> remaining capacity
    capacity = [{} for _ in range(n)]

    def add_edge(u, v, cap):
        if v in capacity[u]:
            capacity[u][v] += cap
        else:
            capacity[u][v] = cap
        if u not in capacity[v]:
            capacity[v][u] = 0

    for i in range(n):
        j1 = i + 1
        if j1 < n:
            add_edge(i, j1, i % 5 + 1)
        j2 = (i + n // 3) % n
        if j2 != i:
            add_edge(i, j2, i % 3 + 1)

    source = 0
    sink = n - 1
    total_flow = 0

    while True:
        # BFS to find augmenting path
        parent = [-1] * n
        parent[source] = source
        queue = [source]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if u == sink:
                break
            for v, cap in capacity[u].items():
                if cap > 0 and parent[v] == -1:
                    parent[v] = u
                    queue.append(v)

        if parent[sink] == -1:
            break

        # Find bottleneck
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            if capacity[u][v] < path_flow:
                path_flow = capacity[u][v]
            v = u

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = u

        total_flow += path_flow

    return total_flow
