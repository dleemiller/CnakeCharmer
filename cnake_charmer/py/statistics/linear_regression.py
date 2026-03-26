"""Simple linear regression: compute slope from deterministic points.

Keywords: statistics, linear regression, slope, intercept, least squares, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def linear_regression(n: int) -> float:
    """Compute slope of simple linear regression on n deterministic points.

    x[i] = i, y[i] = (i*17 + 5) % 1000 / 10.0

    Uses the standard least-squares formula:
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

    Args:
        n: Number of data points.

    Returns:
        The slope of the regression line.
    """
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i in range(n):
        x = float(i)
        y = ((i * 17 + 5) % 1000) / 10.0
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom
