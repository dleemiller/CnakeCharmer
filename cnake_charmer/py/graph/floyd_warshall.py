"""Floyd-Warshall all-pairs shortest paths.

Keywords: graph, floyd-warshall, shortest path, all-pairs, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def floyd_warshall(n: int) -> int:
    """Compute all-pairs shortest paths on n nodes.

    Edge weights: w[i][(i*3+1)%n] = (i%10+1).
    Uses Floyd-Warshall algorithm with O(n^3) triple loop.
    Returns sum of all finite distances.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (total distance sum, dist[0][n-1], max finite distance).
    """
    INF = 10**9

    # Initialize distance matrix (flat)
    dist = [INF] * (n * n)
    for i in range(n):
        dist[i * n + i] = 0
        j = (i * 3 + 1) % n
        w = (i % 10) + 1
        if w < dist[i * n + j]:
            dist[i * n + j] = w

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            ik = dist[i * n + k]
            if ik == INF:
                continue
            for j in range(n):
                new_dist = ik + dist[k * n + j]
                if new_dist < dist[i * n + j]:
                    dist[i * n + j] = new_dist

    dist_0_to_last = dist[0 * n + n - 1]
    total = 0
    max_dist = 0
    for i in range(n * n):
        if dist[i] < INF:
            total += dist[i]
            if dist[i] > max_dist:
                max_dist = dist[i]
    return (total, dist_0_to_last, max_dist)
