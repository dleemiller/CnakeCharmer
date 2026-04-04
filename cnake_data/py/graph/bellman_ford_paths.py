"""Bellman-Ford shortest paths with negative edge detection.

Keywords: graph, bellman-ford, shortest path, negative cycle, relaxation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def bellman_ford_paths(n: int) -> tuple:
    """Compute shortest paths from node 0 using Bellman-Ford.

    Graph has all positive weights. Includes negative cycle detection check.
    Edge weights are deterministic based on node index.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (dist_to_last, negative_cycle_found, dist_to_mid).
    """
    INF = 10**18

    # Build edge list: 3 edges per node, some with negative weights
    num_edges = n * 3
    edges_u = [0] * num_edges
    edges_v = [0] * num_edges
    edges_w = [0] * num_edges

    for i in range(n):
        edges_u[i * 3] = i
        edges_v[i * 3] = (i * 3 + 1) % n
        edges_w[i * 3] = (i % 10) + 1

        edges_u[i * 3 + 1] = i
        edges_v[i * 3 + 1] = (i * 7 + 2) % n
        edges_w[i * 3 + 1] = (i % 5) + 2

        edges_u[i * 3 + 2] = i
        edges_v[i * 3 + 2] = (i * 11 + 5) % n
        edges_w[i * 3 + 2] = (i % 4) + 1

    dist = [INF] * n
    dist[0] = 0

    for _iteration in range(n - 1):
        updated = False
        for e in range(num_edges):
            u = edges_u[e]
            v = edges_v[e]
            w = edges_w[e]
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Check for negative cycles
    negative_cycle_found = 0
    for e in range(num_edges):
        u = edges_u[e]
        v = edges_v[e]
        w = edges_w[e]
        if dist[u] < INF and dist[u] + w < dist[v]:
            negative_cycle_found = 1
            break

    mid = n // 2
    dist_to_last = dist[n - 1] if dist[n - 1] < INF else -1
    dist_to_mid = dist[mid] if dist[mid] < INF else -1

    return (dist_to_last, negative_cycle_found, dist_to_mid)
