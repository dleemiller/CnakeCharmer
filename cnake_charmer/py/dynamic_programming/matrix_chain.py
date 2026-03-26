"""
Minimum scalar multiplications for matrix chain multiplication.

Keywords: dynamic programming, matrix chain, optimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def matrix_chain(n: int) -> int:
    """Find minimum number of scalar multiplications to multiply a chain of n matrices.

    Matrix dimensions: d[i] = 10 + (i*7+3)%90 for i in range(n+1).
    Uses classic O(n^3) DP.

    Args:
        n: Number of matrices in the chain.

    Returns:
        Minimum number of scalar multiplications.
    """
    # Generate dimensions: matrix i has dimensions d[i] x d[i+1]
    d = [10 + (i * 7 + 3) % 90 for i in range(n + 1)]

    # dp[i][j] = minimum cost to multiply matrices i..j
    # Using flat array: dp[i*n + j]
    dp = [0] * (n * n)

    # Chain length l from 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i * n + j] = 2**62  # large sentinel
            for k in range(i, j):
                cost = dp[i * n + k] + dp[(k + 1) * n + j] + d[i] * d[k + 1] * d[j + 1]
                if cost < dp[i * n + j]:
                    dp[i * n + j] = cost

    return dp[0 * n + (n - 1)]
