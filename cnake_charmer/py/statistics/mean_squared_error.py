"""
Compute mean squared error between predicted and actual values.

Generates n deterministic prediction/actual pairs and computes MSE
plus the maximum absolute error across all pairs.

Keywords: statistics, mean squared error, MSE, regression, error metric, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def mean_squared_error(n: int) -> tuple:
    """Compute MSE and max absolute error for n prediction/actual pairs.

    Predictions: pred[i] = ((i * 31 + 17) % 10007) / 100.0
    Actuals:     actual[i] = ((i * 37 + 23) % 10007) / 100.0

    Args:
        n: Number of data points.

    Returns:
        Tuple of (mse, max_abs_error).
    """
    sum_sq_err = 0.0
    max_abs_err = 0.0

    for i in range(n):
        pred = ((i * 31 + 17) % 10007) / 100.0
        actual = ((i * 37 + 23) % 10007) / 100.0
        diff = pred - actual
        sq_err = diff * diff
        sum_sq_err = sum_sq_err + sq_err

        abs_err = -diff if diff < 0.0 else diff

        if abs_err > max_abs_err:
            max_abs_err = abs_err

    mse = sum_sq_err / n
    return (mse, max_abs_err)
