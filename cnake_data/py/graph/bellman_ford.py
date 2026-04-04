"""Bellman-Ford single-source shortest path from node 0.

Keywords: graph, bellman-ford, shortest path, weighted, relaxation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def bellman_ford(n: int) -> int:
    """Compute sum of shortest path distances from node 0 using Bellman-Ford.

    Builds a deterministic weighted graph with n nodes. Each node i has edges:
      i -> (i*3+1)%n  weight (i%10+1)
      i -> (i*7+2)%n  weight (i%5+1)

    Args:
        n: Number of nodes in the graph.

    Returns:
        Sum of shortest path distances from node 0 to all reachable nodes.
    """
    INF = 10**18

    # Build edge list
    edges = []
    for i in range(n):
        edges.append((i, (i * 3 + 1) % n, i % 10 + 1))
        edges.append((i, (i * 7 + 2) % n, i % 5 + 1))

    dist = [INF] * n
    dist[0] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    total = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]
    return total
