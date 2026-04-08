# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Logistic map bifurcation diagram computation — Cython implementation."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500, 1000, 200))
def logistic_bifurcation(int n_params, int n_iter, int n_out):
    """Compute logistic map bifurcation diagram.

    Args:
        n_params: Number of mu values in [2.5, 4.0].
        n_iter: Total iterations per mu (including burn-in).
        n_out: Number of orbit points to collect per mu.

    Returns:
        Tuple of (total_sum, first_orbit_point, last_orbit_point).
    """
    cdef double mu_min = 2.5
    cdef double mu_max = 4.0
    cdef int burnout = n_iter - n_out
    cdef double total = 0.0
    cdef double first_point = 0.0
    cdef double last_point = 0.0
    cdef double mu, x
    cdef int i, j
    cdef bint is_first = True

    for i in range(n_params):
        mu = mu_min + i * (mu_max - mu_min) / n_params
        x = 0.5
        for j in range(n_iter):
            x = mu * x * (1.0 - x)
            if j >= burnout:
                total += x
                if is_first:
                    first_point = x
                    is_first = False
                last_point = x

    return (total, first_point, last_point)
