"""Compute chi-squared statistic for n bins.

Observed values: o[i] = (i*7+3) % 50 + 1. Expected value: mean of observed.
Returns the chi-squared test statistic.

Keywords: statistics, chi-squared, hypothesis testing, goodness of fit, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def chi_squared(n: int) -> float:
    """Compute chi-squared statistic for n bins.

    Args:
        n: Number of bins.

    Returns:
        Chi-squared statistic value.
    """
    # Compute observed values and their sum
    obs_sum = 0.0
    for i in range(n):
        obs_sum += (i * 7 + 3) % 50 + 1

    expected = obs_sum / n

    # Compute chi-squared
    chi2 = 0.0
    for i in range(n):
        observed = (i * 7 + 3) % 50 + 1
        diff = observed - expected
        chi2 += diff * diff / expected

    return chi2
