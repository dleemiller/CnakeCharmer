# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Log-likelihood computation for a linear model (Cython-optimized).

Model: model[i] = a * x[i] + b + c * ts[i].
Log-likelihood: sum of -(y[i] - model[i])^2 / yerr[i]^2.
Also computes chi-squared and BIC.

Keywords: statistics, log-likelihood, linear model, chi-squared, BIC, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

from libc.math cimport log
from libc.stdlib cimport malloc, free


@cython_benchmark(syntax="cy", args=(100000, 2.5, 0.3, -0.1))
def log_likelihood_linear(int n, double a, double b, double c):
    """Compute log-likelihood, chi-squared, and BIC for a linear model."""
    cdef int i
    cdef double model_i, residual, chi2_term
    cdef double log_lik, chi2, bic
    cdef double *ts = <double *>malloc(n * sizeof(double))
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *y = <double *>malloc(n * sizeof(double))
    cdef double *yerr = <double *>malloc(n * sizeof(double))

    if not ts or not x or not y or not yerr:
        free(ts)
        free(x)
        free(y)
        free(yerr)
        raise MemoryError("Failed to allocate arrays")

    # Generate deterministic data from index arithmetic
    for i in range(n):
        ts[i] = (i * 17 + 3) % 200 * 0.01
        x[i] = (i * 13 + 7) % 150 * 0.02
        y[i] = 2.0 * x[i] + 0.5 + (-0.05) * ts[i] + ((i * 31 + 11) % 100 - 50) * 0.001
        yerr[i] = 0.1 + (i % 10) * 0.01

    # Compute log-likelihood and chi-squared
    log_lik = 0.0
    chi2 = 0.0
    for i in range(n):
        model_i = a * x[i] + b + c * ts[i]
        residual = y[i] - model_i
        chi2_term = (residual * residual) / (yerr[i] * yerr[i])
        log_lik -= chi2_term
        chi2 += chi2_term

    free(ts)
    free(x)
    free(y)
    free(yerr)

    # BIC = chi2 + k * ln(n), k=3 parameters
    bic = chi2 + 3.0 * log(<double>n)

    return (log_lik, chi2, bic)
