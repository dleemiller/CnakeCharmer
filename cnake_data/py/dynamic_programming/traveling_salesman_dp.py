"""
Traveling salesman problem using bitmask dynamic programming.

Keywords: dynamic programming, tsp, traveling salesman, bitmask, optimization, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(16,))
def traveling_salesman_dp(n: int) -> int:
    """Solve TSP using bitmask DP on n cities.

    Distances: d[i][j] = abs(sin(i*0.7) - sin(j*0.7)) * 100
                        + abs(cos(i*1.3) - cos(j*1.3)) * 100.

    Args:
        n: Number of cities.

    Returns:
        Minimum tour cost as int (rounded down).
    """
    if n <= 1:
        return 0

    # Build distance matrix
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = int(
                abs(math.sin(i * 0.7) - math.sin(j * 0.7)) * 100
                + abs(math.cos(i * 1.3) - math.cos(j * 1.3)) * 100
            )

    # dp[mask][i] = minimum cost to visit cities in mask, ending at city i
    INF = 10**9
    full_mask = (1 << n) - 1
    dp = [[INF] * n for _ in range(full_mask + 1)]
    dp[1][0] = 0  # Start at city 0

    for mask in range(1, full_mask + 1):
        for u in range(n):
            if dp[mask][u] >= INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                cost = dp[mask][u] + dist[u][v]
                if cost < dp[new_mask][v]:
                    dp[new_mask][v] = cost

    # Find minimum cost to return to city 0
    result = INF
    for u in range(n):
        cost = dp[full_mask][u] + dist[u][0]
        if cost < result:
            result = cost

    return result
