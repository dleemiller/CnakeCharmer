# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D fluid simulation using Burgers' equation with upwind scheme (Cython-optimized).

Keywords: simulation, fluid, burgers equation, upwind, PDE, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def fluid_1d(int n):
    """Simulate 1D Burgers' equation on n cells for 500 steps with C arrays."""
    cdef int steps = 500
    cdef double dt = 0.001
    cdef double dx = 1.0 / n
    cdef int i, t
    cdef double du, total
    cdef int im1, ip1

    cdef double *u = <double *>malloc(n * sizeof(double))
    cdef double *u_new = <double *>malloc(n * sizeof(double))
    cdef double *tmp

    if not u or not u_new:
        free(u); free(u_new)
        raise MemoryError()

    for i in range(n):
        u[i] = sin(2.0 * M_PI * i / n)

    for t in range(steps):
        for i in range(n):
            im1 = (i - 1 + n) % n
            ip1 = (i + 1) % n
            if u[i] >= 0:
                du = u[i] - u[im1]
            else:
                du = u[ip1] - u[i]
            u_new[i] = u[i] - dt * u[i] * du / dx
        tmp = u
        u = u_new
        u_new = tmp

    total = 0.0
    for i in range(n):
        total += u[i]

    free(u)
    free(u_new)
    return total
