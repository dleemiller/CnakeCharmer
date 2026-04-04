# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute mean squared error between predicted and actual values.

Generates n deterministic prediction/actual pairs and computes MSE
plus the maximum absolute error across all pairs.

Keywords: statistics, mean squared error, MSE, regression, error metric, cython, benchmark
"""

from libc.math cimport fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def mean_squared_error(int n):
    """Compute MSE and max absolute error for n prediction/actual pairs."""
    cdef double sum_sq_err = 0.0
    cdef double max_abs_err = 0.0
    cdef double pred, actual, diff, sq_err, abs_err, mse
    cdef int i

    with nogil:
        for i in range(n):
            pred = ((i * 31 + 17) % 10007) / 100.0
            actual = ((i * 37 + 23) % 10007) / 100.0
            diff = pred - actual
            sq_err = diff * diff
            sum_sq_err = sum_sq_err + sq_err

            abs_err = fabs(diff)
            if abs_err > max_abs_err:
                max_abs_err = abs_err

        mse = sum_sq_err / n

    return (mse, max_abs_err)
