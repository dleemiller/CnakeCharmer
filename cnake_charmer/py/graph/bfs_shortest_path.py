"""
BFS shortest path distances from node 0 on a deterministic graph.

Keywords: graph, bfs, shortest path, breadth-first search, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def bfs_shortest_path(n: int) -> int:
    """Compute sum of shortest path distances from node 0 to all reachable nodes.

    Builds a deterministic graph with n nodes. Each node i has edges to
    (i*3+1)%n, (i*7+2)%n, and (i*11+3)%n. Runs BFS from node 0 and returns
    the sum of all finite distances.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (total distance sum, max distance, reachable node count).
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i * 3 + 1) % n)
        adj[i].append((i * 7 + 2) % n)
        adj[i].append((i * 11 + 3) % n)

    # BFS from node 0
    dist = [-1] * n
    dist[0] = 0
    queue = [0]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)

    # Sum all finite distances
    total = 0
    max_dist = 0
    reachable_count = 0
    for i in range(n):
        if dist[i] != -1:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            reachable_count += 1
    return (total, max_dist, reachable_count)
