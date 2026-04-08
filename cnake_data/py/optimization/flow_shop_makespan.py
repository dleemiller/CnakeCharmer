"""Permutation flow shop mean completion time (Johnson-style scheduling).

Computes the mean job completion time (mean flowtime) for a permutation flow
shop schedule: n jobs × m machines, given a fixed processing time matrix.

Keywords: optimization, scheduling, flow shop, makespan, permutation, combinatorial
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(40, 8))
def flow_shop_makespan(n: int, m: int) -> tuple:
    """Compute mean job completion time for a permutation flow shop.

    Generates deterministic processing times from (n, m) via sin/cos,
    then evaluates the identity permutation [1, 2, ..., n].

    Args:
        n: Number of jobs.
        m: Number of machines.

    Returns:
        Tuple of (mean_completion, first_job_completion, last_job_completion).
    """
    # Generate deterministic processing times in [1, 20]
    orders = [
        [int(10.0 + 9.0 * math.sin(i * 1.3 + j * 0.7)) + 1 for j in range(m)] for i in range(n)
    ]

    # Identity permutation (1-indexed)
    sol = list(range(1, n + 1))

    # Allocate completion-time matrix
    mat = [[0] * m for _ in range(n)]

    # First job
    idx = sol[0] - 1
    c = 0
    for j in range(m):
        c += orders[idx][j]
        mat[idx][j] = c
    total = float(mat[idx][m - 1])

    # Remaining jobs
    for i in range(1, n):
        idx = sol[i] - 1
        prev = sol[i - 1] - 1
        mat[idx][0] = mat[prev][0] + orders[idx][0]
        for j in range(1, m):
            mat[idx][j] = max(mat[prev][j], mat[idx][j - 1]) + orders[idx][j]
        total += mat[idx][m - 1]

    first_completion = mat[sol[0] - 1][m - 1]
    last_completion = mat[sol[-1] - 1][m - 1]
    return (total / n, first_completion, last_completion)
