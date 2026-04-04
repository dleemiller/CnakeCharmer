# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y'=-y+sin(t) from t=0 to t=10 using forward Euler with n steps (Cython-optimized).

Initial condition y(0) = 1. Returns the final value of y.

Keywords: ODE, Euler method, differential equation, forward Euler, numerical, cython
"""

from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def euler_method(int n):
    """Solve y'=-y+sin(t) using forward Euler and return final y value."""
    cdef int i
    cdef double t, y, dt

    t = 0.0
    y = 1.0
    dt = 10.0 / n

    for i in range(n):
        y += dt * (-y + sin(t))
        t += dt

    return y
