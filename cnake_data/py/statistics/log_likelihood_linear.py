"""Log-likelihood computation for a linear model.

Model: model[i] = a * x[i] + b + c * ts[i].
Log-likelihood: sum of -(y[i] - model[i])^2 / yerr[i]^2.
Also computes chi-squared and BIC.

Keywords: statistics, log-likelihood, linear model, chi-squared, BIC, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000, 2.5, 0.3, -0.1))
def log_likelihood_linear(n: int, a: float, b: float, c: float) -> tuple:
    """Compute log-likelihood, chi-squared, and BIC for a linear model.

    Args:
        n: Number of data points.
        a: Coefficient for x.
        b: Intercept.
        c: Coefficient for time series t.

    Returns:
        Tuple of (log_likelihood, chi2, bic).
    """
    # Generate deterministic data from index arithmetic
    log_lik = 0.0
    chi2 = 0.0
    for i in range(n):
        ts_i = (i * 17 + 3) % 200 * 0.01
        x_i = (i * 13 + 7) % 150 * 0.02
        y_i = 2.0 * x_i + 0.5 + (-0.05) * ts_i + ((i * 31 + 11) % 100 - 50) * 0.001
        yerr_i = 0.1 + (i % 10) * 0.01

        model_i = a * x_i + b + c * ts_i
        residual = y_i - model_i
        chi2_term = (residual * residual) / (yerr_i * yerr_i)
        log_lik -= chi2_term
        chi2 += chi2_term

    # BIC = chi2 + k * ln(n), k=3 parameters
    import math

    bic = chi2 + 3.0 * math.log(n)

    return (log_lik, chi2, bic)
