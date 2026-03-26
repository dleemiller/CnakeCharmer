"""Optimal binary search tree cost via dynamic programming.

Keywords: optimal BST, binary search tree, dynamic programming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def optimal_bst(n: int) -> int:
    """Compute the minimum cost of an optimal binary search tree for n keys.

    Frequencies: f[i] = (i*7 + 3) % 100 + 1. Uses O(n^3) DP.

    Args:
        n: Number of keys.

    Returns:
        Minimum search cost as an integer.
    """
    freq = [(i * 7 + 3) % 100 + 1 for i in range(n)]

    # prefix_sum[i] = sum of freq[0..i-1]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    # cost[i][j] = min cost for keys i..j (flat array)
    cost = [0] * (n * n)

    # Base case: single keys
    for i in range(n):
        cost[i * n + i] = freq[i]

    # Fill for lengths 2..n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            weight = prefix[j + 1] - prefix[i]
            best = 10**18
            for r in range(i, j + 1):
                left_cost = cost[i * n + (r - 1)] if r > i else 0
                right_cost = cost[(r + 1) * n + j] if r < j else 0
                val = left_cost + right_cost
                if val < best:
                    best = val
            cost[i * n + j] = best + weight

    return cost[0 * n + (n - 1)]
