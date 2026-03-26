# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y'=y*cos(t) from t=0 to t=5 using midpoint method with n steps (Cython-optimized).

Initial condition y(0) = 1. Returns the final value of y.

Keywords: ODE, midpoint method, differential equation, numerical, second-order, cython
"""

from libc.math cimport cos
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def midpoint_method(int n):
    """Solve y'=y*cos(t) using midpoint method and return final y value."""
    cdef int i
    cdef double t, y, dt, k1, y_mid, t_mid, k2

    t = 0.0
    y = 1.0
    dt = 5.0 / n

    for i in range(n):
        k1 = y * cos(t)
        y_mid = y + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = y_mid * cos(t_mid)
        y += dt * k2
        t += dt

    return y
