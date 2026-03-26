# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve 1D advection equation du/dt + du/dx = 0 using method of lines (Cython-optimized).

Spatial finite differences + RK4 in time on n cells for 1000 steps.
u(x,0) = exp(-((x-0.5)/0.1)^2). Return sum of final u.

Keywords: PDE, advection, method of lines, RK4, finite difference, numerical, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def method_of_lines(int n):
    """Solve 1D advection equation using method of lines."""
    cdef int i, step
    cdef int num_steps = 1000
    cdef double dx, dt, x, arg, inv_dx, total, dt6
    cdef double *u
    cdef double *k1
    cdef double *k2
    cdef double *k3
    cdef double *k4
    cdef double *tmp

    dx = 1.0 / n
    dt = 0.5 * dx
    inv_dx = 1.0 / dx
    dt6 = dt / 6.0

    u = <double *>malloc(n * sizeof(double))
    k1 = <double *>malloc(n * sizeof(double))
    k2 = <double *>malloc(n * sizeof(double))
    k3 = <double *>malloc(n * sizeof(double))
    k4 = <double *>malloc(n * sizeof(double))
    tmp = <double *>malloc(n * sizeof(double))
    if not u or not k1 or not k2 or not k3 or not k4 or not tmp:
        free(u)
        free(k1)
        free(k2)
        free(k3)
        free(k4)
        free(tmp)
        raise MemoryError()

    # Initial condition
    for i in range(n):
        x = i * dx
        arg = (x - 0.5) / 0.1
        u[i] = exp(-arg * arg)

    for step in range(num_steps):
        # k1 = rhs(u)
        k1[0] = -(u[0] - u[n - 1]) * inv_dx
        for i in range(1, n):
            k1[i] = -(u[i] - u[i - 1]) * inv_dx

        # k2 = rhs(u + 0.5*dt*k1)
        for i in range(n):
            tmp[i] = u[i] + 0.5 * dt * k1[i]
        k2[0] = -(tmp[0] - tmp[n - 1]) * inv_dx
        for i in range(1, n):
            k2[i] = -(tmp[i] - tmp[i - 1]) * inv_dx

        # k3 = rhs(u + 0.5*dt*k2)
        for i in range(n):
            tmp[i] = u[i] + 0.5 * dt * k2[i]
        k3[0] = -(tmp[0] - tmp[n - 1]) * inv_dx
        for i in range(1, n):
            k3[i] = -(tmp[i] - tmp[i - 1]) * inv_dx

        # k4 = rhs(u + dt*k3)
        for i in range(n):
            tmp[i] = u[i] + dt * k3[i]
        k4[0] = -(tmp[0] - tmp[n - 1]) * inv_dx
        for i in range(1, n):
            k4[i] = -(tmp[i] - tmp[i - 1]) * inv_dx

        # Update
        for i in range(n):
            u[i] += dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    total = 0.0
    for i in range(n):
        total += u[i]

    free(u)
    free(k1)
    free(k2)
    free(k3)
    free(k4)
    free(tmp)
    return total
