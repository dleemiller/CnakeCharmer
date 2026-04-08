# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Lotka-Volterra three-species predator-prey ODE solved with RK4 — Cython implementation."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def lotka_volterra_rk4(int n_steps):
    """Integrate a three-species Lotka-Volterra system with RK4.

    Args:
        n_steps: Number of integration steps.

    Returns:
        Tuple of (mean_s, mean_z, mean_r) — time-averaged values over the trajectory.
    """
    cdef double alpha = 1.0
    cdef double beta = 0.1
    cdef double delta = 0.05
    cdef double xi = 0.075
    cdef double alpha2 = 0.04
    cdef double sigma = 0.2
    cdef double h = 0.1

    cdef double s = 10.0
    cdef double z = 5.0
    cdef double r = 2.0

    cdef double sum_s = 0.0
    cdef double sum_z = 0.0
    cdef double sum_r = 0.0

    cdef double k1, k2, k3, k4
    cdef double l1, l2, l3, l4
    cdef double m1, m2, m3, m4
    cdef double s2, z2, r2
    cdef int i

    for i in range(n_steps):
        # k1 / l1 / m1
        k1 = h * (alpha * s - beta * s * z - delta * s)
        l1 = h * (beta * s * z + xi * r - alpha2 * s * z - sigma * z)
        m1 = h * (delta * s + alpha2 * s * z - xi * r + sigma * z)

        # k2 / l2 / m2  (midpoint using k1 increments)
        s2 = s + k1 * 0.5
        z2 = z + l1 * 0.5
        r2 = r + m1 * 0.5
        k2 = h * (alpha * s2 - beta * s2 * z2 - delta * s2)
        l2 = h * (beta * s2 * z2 + xi * r2 - alpha2 * s2 * z2 - sigma * z2)
        m2 = h * (delta * s2 + alpha2 * s2 * z2 - xi * r2 + sigma * z2)

        # k3 / l3 / m3  (midpoint using k2 increments)
        s2 = s + k2 * 0.5
        z2 = z + l2 * 0.5
        r2 = r + m2 * 0.5
        k3 = h * (alpha * s2 - beta * s2 * z2 - delta * s2)
        l3 = h * (beta * s2 * z2 + xi * r2 - alpha2 * s2 * z2 - sigma * z2)
        m3 = h * (delta * s2 + alpha2 * s2 * z2 - xi * r2 + sigma * z2)

        # k4 / l4 / m4  (endpoint using k3 increments)
        s2 = s + k3
        z2 = z + l3
        r2 = r + m3
        k4 = h * (alpha * s2 - beta * s2 * z2 - delta * s2)
        l4 = h * (beta * s2 * z2 + xi * r2 - alpha2 * s2 * z2 - sigma * z2)
        m4 = h * (delta * s2 + alpha2 * s2 * z2 - xi * r2 + sigma * z2)

        s = s + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z = z + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
        r = r + (m1 + 2.0 * m2 + 2.0 * m3 + m4) / 6.0

        if s < 0.0:
            s = 0.0
        if z < 0.0:
            z = 0.0
        if r < 0.0:
            r = 0.0

        sum_s += s
        sum_z += z
        sum_r += r

    return (sum_s / n_steps, sum_z / n_steps, sum_r / n_steps)
