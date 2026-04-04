# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y'=-2*y+t using 4-step Adams-Bashforth method (Cython-optimized).

From t=0 to t=5, y(0)=1. Bootstrap with RK4. Returns final y.

Keywords: ODE, Adams-Bashforth, multistep method, differential equation, numerical, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def adams_bashforth(int n):
    """Solve y'=-2*y+t using 4-step Adams-Bashforth and return final y."""
    cdef int i
    cdef double dt, t_val, y_val, k1, k2, k3, k4
    cdef double fm0, fm1, fm2, fm3
    cdef double t_hist[4]
    cdef double y_hist[4]
    cdef double f_hist[4]

    dt = 5.0 / n

    # Bootstrap first 4 steps with RK4
    t_hist[0] = 0.0
    y_hist[0] = 1.0
    f_hist[0] = -2.0 * y_hist[0] + t_hist[0]

    for i in range(1, 4):
        t_val = t_hist[i - 1]
        y_val = y_hist[i - 1]
        k1 = dt * (-2.0 * y_val + t_val)
        k2 = dt * (-2.0 * (y_val + 0.5 * k1) + t_val + 0.5 * dt)
        k3 = dt * (-2.0 * (y_val + 0.5 * k2) + t_val + 0.5 * dt)
        k4 = dt * (-2.0 * (y_val + k3) + t_val + dt)
        y_hist[i] = y_val + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t_hist[i] = t_val + dt
        f_hist[i] = -2.0 * y_hist[i] + t_hist[i]

    t_val = t_hist[3]
    y_val = y_hist[3]
    fm3 = f_hist[0]
    fm2 = f_hist[1]
    fm1 = f_hist[2]
    fm0 = f_hist[3]

    for i in range(4, n):
        y_val = y_val + dt / 24.0 * (55.0 * fm0 - 59.0 * fm1 + 37.0 * fm2 - 9.0 * fm3)
        t_val += dt
        fm3 = fm2
        fm2 = fm1
        fm1 = fm0
        fm0 = -2.0 * y_val + t_val

    return y_val
