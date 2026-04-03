# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve du/dt = -u + sin(t) from t=0 to t=10 using RK2 midpoint method (Cython-optimized).

Initial condition u(0) = 0. Returns tuple of (final_u, u_at_midpoint).

Keywords: ODE, RK2, Runge-Kutta, midpoint method, differential equation, numerical, cython
"""

from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def rk2_ode_step(int n):
    """Solve du/dt = -u + sin(t) using RK2 midpoint method."""
    cdef int i, mid_step
    cdef double u, t, dt, f1, f2, K1, K2
    cdef double u_at_mid

    u = 0.0
    t = 0.0
    dt = 10.0 / n
    mid_step = n // 2
    u_at_mid = 0.0

    with nogil:
        for i in range(n):
            # f(u, t) = -u + sin(t)
            f1 = -u + sin(t)
            K1 = dt * f1
            # f(u + 0.5*K1, t + 0.5*dt)
            f2 = -(u + 0.5 * K1) + sin(t + 0.5 * dt)
            K2 = dt * f2
            u = u + K2
            t += dt

            if i == mid_step:
                u_at_mid = u

    return (u, u_at_mid)
