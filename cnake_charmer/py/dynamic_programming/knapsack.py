"""
0/1 Knapsack problem.

Keywords: dynamic programming, knapsack, optimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def knapsack(n: int) -> int:
    """Solve the 0/1 knapsack problem with n items.

    Weights = i%10+1, values = i*3%17+1 for each item i.
    Capacity = n*3.

    Args:
        n: Number of items.

    Returns:
        Maximum achievable value.
    """
    capacity = n * 3
    weights = [i % 10 + 1 for i in range(n)]
    values = [i * 3 % 17 + 1 for i in range(n)]

    # 1D DP array
    dp = [0] * (capacity + 1)

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(capacity, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v

    return (dp[capacity], dp[capacity // 2])
