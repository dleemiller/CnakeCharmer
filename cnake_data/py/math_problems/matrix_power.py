"""Compute sum of traces of matrix powers M^1 + M^2 + ... + M^n.

Keywords: matrix, exponentiation, trace, linear algebra, modular arithmetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(100000,))
def matrix_power(n: int) -> int:
    """Compute sum of traces of M^1 through M^n mod 10^9+7.

    M is the fixed 3x3 matrix [[1,1,0],[1,0,1],[0,1,1]].
    Uses repeated matrix multiplication (not fast exponentiation in the
    Python version for simplicity -- accumulates M^k iteratively).

    Args:
        n: Number of matrix powers to sum traces of.

    Returns:
        Sum of tr(M^1) + tr(M^2) + ... + tr(M^n), mod 10^9+7.
    """
    mod = MOD

    # M = [[1,1,0],[1,0,1],[0,1,1]]
    # Current power of M (start with M^1 = M)
    cur = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]

    total = 0
    for _ in range(n):
        # Add trace of current power
        trace = (cur[0][0] + cur[1][1] + cur[2][2]) % mod
        total = (total + trace) % mod

        # Multiply cur by M
        new = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                s = 0
                for k in range(3):
                    s += (
                        cur[i][k]
                        * [
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                        ][k][j]
                    )
                new[i][j] = s % mod
        cur = new

    return total
