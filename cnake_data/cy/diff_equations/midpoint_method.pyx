# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y'=y*cos(t) from t=0 to t=5 using midpoint method (Cython-optimized).

Initial condition y(0) = 1. Returns tuple of trajectory metrics.

Keywords: ODE, midpoint method, differential equation, numerical, second-order, cython
"""

from libc.math cimport cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def midpoint_method(int n):
    """Solve y'=y*cos(t) using midpoint method and return trajectory metrics."""
    cdef int i, mid_step
    cdef double t, y, dt, k1, y_mid_val, t_mid, k2
    cdef double max_y, y_at_mid

    t = 0.0
    y = 1.0
    dt = 5.0 / n
    mid_step = n // 2
    max_y = y
    y_at_mid = y

    for i in range(n):
        k1 = y * cos(t)
        y_mid_val = y + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = y_mid_val * cos(t_mid)
        y += dt * k2
        t += dt

        if y > max_y:
            max_y = y
        if i == mid_step:
            y_at_mid = y

    return (y, y_at_mid, max_y)
