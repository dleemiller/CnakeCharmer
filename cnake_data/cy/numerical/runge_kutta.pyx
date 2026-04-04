# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y'=y*sin(t) from t=0 to t=10 using RK4 with n steps (Cython-optimized).

Initial condition y(0) = 1. Returns the final value of y.

Keywords: numerical, ODE, Runge-Kutta, RK4, differential equation, cython, benchmark
"""

from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def runge_kutta(int n):
    """Solve y'=y*sin(t) using RK4 and return final y value."""
    cdef int i
    cdef double t, y, dt, k1, k2, k3, k4, half_dt, t_half

    t = 0.0
    y = 1.0
    dt = 10.0 / n
    half_dt = 0.5 * dt

    for i in range(n):
        t_half = t + half_dt
        k1 = dt * y * sin(t)
        k2 = dt * (y + 0.5 * k1) * sin(t_half)
        k3 = dt * (y + 0.5 * k2) * sin(t_half)
        k4 = dt * (y + k3) * sin(t + dt)
        y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += dt

    return y
