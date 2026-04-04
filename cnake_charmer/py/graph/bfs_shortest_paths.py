"""
BFS shortest path distances from node 0 in a sparse deterministic graph.

Keywords: graph, bfs, shortest path, breadth-first search, benchmark
"""

from collections import deque

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def bfs_shortest_paths(n: int) -> tuple:
    """BFS shortest-path distances from node 0 in a deterministic sparse graph.

    Graph construction: for each i in range(n), add edges
    (i, (i+1)%n) and (i, (i*3+7)%n).

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (sum_of_distances % 10**9, max_distance, num_reachable).
    """
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[i].append((i * 3 + 7) % n)

    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)

    total = 0
    max_dist = 0
    num_reachable = 0
    for i in range(n):
        if dist[i] != -1:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            num_reachable += 1

    return (total % (10**9), max_dist, num_reachable)
