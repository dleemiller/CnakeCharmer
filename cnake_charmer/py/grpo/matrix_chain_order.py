"""Optimal matrix chain multiplication order via dynamic programming.

Keywords: grpo, dynamic programming, matrix, optimization, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def matrix_chain_order(n: int) -> tuple:
    """Find minimum scalar multiplications for matrix chain product.

    Given n matrices with deterministic dimensions, computes the optimal
    parenthesization cost using standard O(n^3) DP.

    Returns (min_cost, number of DP cells filled, trace checksum).

    Args:
        n: Number of matrices in the chain.

    Returns:
        Tuple of (min_cost, cells_filled, checksum).
    """
    # Generate deterministic dimensions: matrix i has dims[i] x dims[i+1]
    dims = [0] * (n + 1)
    for i in range(n + 1):
        dims[i] = 10 + (i * 37 + 13) % 90

    # DP table
    m = [[0] * n for _ in range(n)]
    cells = 0

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = 1 << 62  # large number
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                cells += 1

    # Checksum from diagonal
    checksum = 0
    for i in range(n):
        for j in range(i, min(i + 5, n)):
            checksum = (checksum * 31 + m[i][j]) & 0xFFFFFFFF

    return (m[0][n - 1], cells, checksum)
