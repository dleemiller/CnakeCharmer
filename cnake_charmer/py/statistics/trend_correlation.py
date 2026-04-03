"""
Trend-preserving correlation of a deterministic matrix.

For each row, compute trend signs (direction of consecutive differences: +1 or -1).
Then compute pairwise Hamming correlation between all row pairs and average.

Keywords: statistics, trend, correlation, hamming, pairwise, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def trend_correlation(n: int) -> tuple:
    """Compute trend-preserving correlation of an n x 15 deterministic matrix.

    For each row, compute a trend vector of signs of consecutive differences.
    For all pairs (i < j), compute Hamming correlation: max(same/total, diff/total).
    Return average correlation and sum of all trend values.

    Args:
        n: Number of rows in the matrix.

    Returns:
        Tuple of (average_correlation, trend_sum).
    """
    cols = 15
    trend_len = cols - 1

    # Build the matrix and trend vectors
    trends = []
    trend_sum = 0.0
    for i in range(n):
        row_trends = [0.0] * trend_len
        for j in range(trend_len):
            val_cur = ((i * 17 + j * 31 + 5) % 1000) / 100.0
            val_nxt = ((i * 17 + (j + 1) * 31 + 5) % 1000) / 100.0
            if val_nxt - val_cur > 0:
                row_trends[j] = 1.0
            else:
                row_trends[j] = -1.0
            trend_sum += row_trends[j]
        trends.append(row_trends)

    # Pairwise Hamming correlation
    total_corr = 0.0
    pair_count = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            same = 0.0
            diff = 0.0
            for k in range(trend_len):
                if trends[i][k] == trends[j][k]:
                    same += 1.0
                else:
                    diff += 1.0
            ratio_same = same / trend_len
            ratio_diff = diff / trend_len
            if ratio_same > ratio_diff:
                total_corr += ratio_same
            else:
                total_corr += ratio_diff
            pair_count += 1

    avg_corr = total_corr / pair_count
    return (avg_corr, trend_sum)
