"""Floyd-Warshall all-pairs shortest paths with path counting.

Keywords: graph, floyd-warshall, all-pairs, shortest path, path counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(400,))
def floyd_warshall_apsp(n: int) -> tuple:
    """Compute all-pairs shortest paths and count finite paths.

    Graph: node i -> (i+1)%n weight (i%5+1), i -> (i*3+1)%n weight (i%7+2),
    i -> (i*5+2)%n weight (i%4+3). Uses flat array for distance matrix.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (dist_0_to_last, dist_mid_to_last, total_finite_paths).
    """
    INF = 10**9

    # Initialize distance matrix (flat)
    size = n * n
    dist = [INF] * size
    for i in range(n):
        dist[i * n + i] = 0
        # Edge to next node (ensures connectivity)
        j0 = (i + 1) % n
        w0 = (i % 5) + 1
        if w0 < dist[i * n + j0]:
            dist[i * n + j0] = w0
        j1 = (i * 3 + 1) % n
        w1 = (i % 7) + 2
        if w1 < dist[i * n + j1]:
            dist[i * n + j1] = w1
        j2 = (i * 5 + 2) % n
        w2 = (i % 4) + 3
        if w2 < dist[i * n + j2]:
            dist[i * n + j2] = w2

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

    mid = n // 2
    dist_0_to_last = dist[0 * n + n - 1]
    dist_mid_to_last = dist[mid * n + n - 1]

    total_finite_paths = 0
    for i in range(size):
        if dist[i] < INF:
            total_finite_paths += 1

    return (dist_0_to_last, dist_mid_to_last, total_finite_paths)
