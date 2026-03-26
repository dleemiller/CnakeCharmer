"""Spearman rank correlation coefficient.

Keywords: statistics, spearman, correlation, ranking, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def spearman_correlation(n: int) -> float:
    """Compute Spearman rank correlation of two sequences of length n.

    x[i] = (i*7+3) % 1000, y[i] = (i*13+7) % 1000.
    Ranks are assigned by sorting; ties get average rank.

    Args:
        n: Length of sequences.

    Returns:
        Spearman rank correlation coefficient.
    """
    import math

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

    x = [(i * 7 + 3) % 1000 for i in range(n)]
    y = [(i * 13 + 7) % 1000 for i in range(n)]

    rx = rank_data(x)
    ry = rank_data(y)

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

    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)
