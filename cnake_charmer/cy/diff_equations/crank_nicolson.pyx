# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve 1D heat equation using Crank-Nicolson scheme (Cython-optimized).

n cells, 1000 time steps. u(x,0) = sin(pi*x/n)*100. Returns sum of final values.

Keywords: PDE, heat equation, Crank-Nicolson, tridiagonal, Thomas algorithm, numerical, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def crank_nicolson(int n):
    """Solve 1D heat equation using Crank-Nicolson scheme."""
    cdef int i, step
    cdef int num_steps = 1000
    cdef double dx, dt, r, alpha, beta, gamma
    cdef double alpha_rhs, beta_rhs, gamma_rhs, total
    cdef double *u
    cdef double *cp
    cdef double *dp
    cdef double *rhs

    dx = 1.0 / n
    dt = 0.5 * dx * dx
    r = dt / (dx * dx)

    u = <double *>malloc(n * sizeof(double))
    cp = <double *>malloc(n * sizeof(double))
    dp = <double *>malloc(n * sizeof(double))
    rhs = <double *>malloc(n * sizeof(double))
    if not u or not cp or not dp or not rhs:
        if u:
            free(u)
        if cp:
            free(cp)
        if dp:
            free(dp)
        if rhs:
            free(rhs)
        raise MemoryError()

    # Initial condition
    for i in range(n):
        u[i] = sin(M_PI * i / n) * 100.0

    alpha = -0.5 * r
    beta = 1.0 + r
    gamma = -0.5 * r
    alpha_rhs = 0.5 * r
    beta_rhs = 1.0 - r
    gamma_rhs = 0.5 * r

    for step in range(num_steps):
        # Build RHS
        rhs[0] = beta_rhs * u[0] + gamma_rhs * u[1]
        for i in range(1, n - 1):
            rhs[i] = alpha_rhs * u[i - 1] + beta_rhs * u[i] + gamma_rhs * u[i + 1]
        rhs[n - 1] = alpha_rhs * u[n - 2] + beta_rhs * u[n - 1]

        # Thomas algorithm - forward sweep
        cp[0] = gamma / beta
        dp[0] = rhs[0] / beta
        for i in range(1, n):
            cp[i] = gamma / (beta - alpha * cp[i - 1])
            dp[i] = (rhs[i] - alpha * dp[i - 1]) / (beta - alpha * cp[i - 1])

        # Back substitution
        u[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            u[i] = dp[i] - cp[i] * u[i + 1]

    total = 0.0
    for i in range(n):
        total += u[i]

    free(u)
    free(cp)
    free(dp)
    free(rhs)
    return total
