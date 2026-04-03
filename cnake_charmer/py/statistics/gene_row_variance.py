"""Compute pooled within-row variance of a 2D matrix.

For each row, subtract row mean, square, sum all, divide by total elements.

Keywords: statistics, variance, gene, row, matrix, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(8000,))
def gene_row_variance(n: int) -> tuple:
    """Compute pooled within-row variance of an n x 20 matrix.

    Matrix: val[i][j] = ((i * 7 + j * 13 + 42) % 1000) / 100.0
    For each row compute mean, then sum squared deviations across all rows,
    divide by total element count.

    Args:
        n: Number of rows in the matrix (columns fixed at 20).

    Returns:
        Tuple of (variance, row_means_sum).
    """
    cols = 20
    row_means_sum = 0.0
    total_sq = 0.0

    for i in range(n):
        # Compute row mean
        row_sum = 0.0
        for j in range(cols):
            row_sum += ((i * 7 + j * 13 + 42) % 1000) / 100.0
        row_mean = row_sum / cols
        row_means_sum += row_mean

        # Accumulate squared deviations
        for j in range(cols):
            val = ((i * 7 + j * 13 + 42) % 1000) / 100.0
            diff = val - row_mean
            total_sq += diff * diff

    variance = total_sq / (n * cols)
    return (variance, row_means_sum)
