"""Logistic map bifurcation diagram computation.

Iterates the logistic map x_{n+1} = mu * x * (1 - x) across a range of
mu values, discarding transient iterations and collecting orbit points.

Keywords: logistic map, bifurcation, chaos, nonlinear dynamics, simulation
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500, 1000, 200))
def logistic_bifurcation(n_params: int, n_iter: int, n_out: int) -> tuple:
    """Compute logistic map bifurcation diagram.

    Args:
        n_params: Number of mu values in [2.5, 4.0].
        n_iter: Total iterations per mu (including burn-in).
        n_out: Number of orbit points to collect per mu.

    Returns:
        Tuple of (total_sum, first_orbit_point, last_orbit_point).
    """
    mu_min = 2.5
    mu_max = 4.0
    burnout = n_iter - n_out

    total = 0.0
    first_point = 0.0
    last_point = 0.0
    is_first = True

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
