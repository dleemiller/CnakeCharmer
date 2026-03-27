"""Spearman rank correlation with detailed statistics.

Keywords: statistics, spearman, rank, correlation, d-squared, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def spearman_rank(n: int) -> tuple:
    """Compute Spearman rank correlation with additional statistics.

    x[i] = (i*11+5) % 1009, y[i] = (i*19+3) % 1013.
    Ranks are assigned by sorting; ties get average rank.

    Args:
        n: Length of sequences.

    Returns:
        Tuple of (correlation, sum_d_squared, mean_rank).
    """
    x = [(i * 11 + 5) % 1009 for i in range(n)]
    y = [(i * 19 + 3) % 1013 for i in range(n)]

    def rank_data(vals):
        indexed = sorted(range(len(vals)), key=lambda k: vals[k])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j < len(vals) - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)

    # Compute sum of d^2 and mean rank
    sum_d_sq = 0.0
    mean_rank = 0.0
    for i in range(n):
        d = rx[i] - ry[i]
        sum_d_sq += d * d
        mean_rank += rx[i]
    mean_rank = mean_rank / n

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = rx[i] - mean_rx
        dy = ry[i] - mean_ry
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    correlation = 0.0 if var_x == 0.0 or var_y == 0.0 else cov / math.sqrt(var_x * var_y)

    return (correlation, sum_d_sq, mean_rank)
