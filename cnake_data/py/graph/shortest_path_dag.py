"""Shortest path in a DAG using topological ordering.

Keywords: graph, DAG, shortest path, topological sort, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def shortest_path_dag(n: int) -> tuple:
    """Compute shortest paths from node 0 in a weighted DAG.

    DAG edges: for each node i, edges to (i+1+i%3) and (i+2+i%5) if target < n,
    with weights ((i*7+3)%50)+1 and ((i*11+1)%40)+1 respectively.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (sum of finite distances, max distance, reachable count).
    """
    if n < 1:
        return (0, 0, 0)

    INF = 10**18

    # Build adjacency list (DAG: edges only go to higher-numbered nodes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        t1 = i + 1 + i % 3
        if t1 < n:
            w1 = ((i * 7 + 3) % 50) + 1
            adj[i].append((t1, w1))
        t2 = i + 2 + i % 5
        if t2 < n:
            w2 = ((i * 11 + 1) % 40) + 1
            adj[i].append((t2, w2))

    # Topological order is simply 0, 1, 2, ..., n-1 since edges go forward
    dist = [INF] * n
    dist[0] = 0

    for u in range(n):
        if dist[u] == INF:
            continue
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd

    total = 0
    max_dist = 0
    reachable = 0
    for i in range(n):
        if dist[i] < INF:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
            reachable += 1

    return (total, max_dist, reachable)
