"""
Dijkstra's shortest path from node 0 on a weighted deterministic graph.

Keywords: graph, dijkstra, shortest path, weighted, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def dijkstra(n: int) -> int:
    """Compute sum of shortest path distances from node 0 using Dijkstra's algorithm.

    Builds a deterministic weighted graph with n nodes. Each node i has edges:
      i -> (i*3+1)%n  weight (i%10+1)
      i -> (i*7+2)%n  weight (i%5+1)
      i -> (i*11+3)%n weight (i%7+1)

    Uses a simple O(V^2) Dijkstra (no heap) and returns the sum of all finite
    shortest distances from node 0.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (total distance sum, max distance, reachable node count).
    """
    INF = 10**18

    # Build adjacency list with weights
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append(((i * 3 + 1) % n, i % 10 + 1))
        adj[i].append(((i * 7 + 2) % n, i % 5 + 1))
        adj[i].append(((i * 11 + 3) % n, i % 7 + 1))

    dist = [INF] * n
    visited = [False] * n
    dist[0] = 0

    for _ in range(n):
        # Find unvisited node with minimum distance
        u = -1
        min_d = INF
        for v in range(n):
            if not visited[v] and dist[v] < min_d:
                min_d = dist[v]
                u = v
        if u == -1:
            break
        visited[u] = True
        # Relax edges
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd

    total = 0
    max_dist = 0
    reachable_count = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            reachable_count += 1
    return (total, max_dist, reachable_count)
