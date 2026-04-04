"""Dijkstra shortest path with relaxation counting.

Keywords: graph, dijkstra, shortest path, weighted, relaxation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(4000,))
def dijkstra_shortest(n: int) -> tuple:
    """Compute shortest paths from node 0 on deterministic weighted graph.

    Graph: node i has edges to (i*5+3)%n weight (i%8+1),
    (i*9+7)%n weight (i%6+2), (i*13+11)%n weight (i%4+3).
    Uses O(V^2) Dijkstra. Counts total edge relaxations.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (dist_to_last, dist_to_mid, total_relaxations).
    """
    INF = 10**18

    # Build adjacency list
    adj = [None] * n
    for i in range(n):
        adj[i] = [
            ((i * 5 + 3) % n, i % 8 + 1),
            ((i * 9 + 7) % n, i % 6 + 2),
            ((i * 13 + 11) % n, i % 4 + 3),
        ]

    dist = [INF] * n
    visited = [False] * n
    dist[0] = 0
    total_relaxations = 0

    for _ in range(n):
        u = -1
        min_d = INF
        for v in range(n):
            if not visited[v] and dist[v] < min_d:
                min_d = dist[v]
                u = v
        if u == -1:
            break
        visited[u] = True
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                total_relaxations += 1

    mid = n // 2
    dist_to_last = dist[n - 1] if dist[n - 1] < INF else -1
    dist_to_mid = dist[mid] if dist[mid] < INF else -1

    return (dist_to_last, dist_to_mid, total_relaxations)
