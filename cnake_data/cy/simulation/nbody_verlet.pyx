# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""N-body simulation with Verlet integration (Cython-optimized).

Keywords: n-body, Verlet integration, physics simulation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def nbody_verlet(int n):
    """N-body Verlet integration using C arrays."""
    cdef double dt = 0.001
    cdef double soft2 = 0.01  # softening^2
    cdef int steps = 100

    cdef double *px = <double *>malloc(n * sizeof(double))
    cdef double *py = <double *>malloc(n * sizeof(double))
    cdef double *vx = <double *>malloc(n * sizeof(double))
    cdef double *vy = <double *>malloc(n * sizeof(double))
    cdef double *ax = <double *>malloc(n * sizeof(double))
    cdef double *ay = <double *>malloc(n * sizeof(double))
    cdef double *ax_new = <double *>malloc(n * sizeof(double))
    cdef double *ay_new = <double *>malloc(n * sizeof(double))

    if not px or not py or not vx or not vy or not ax or not ay or not ax_new or not ay_new:
        if px: free(px)
        if py: free(py)
        if vx: free(vx)
        if vy: free(vy)
        if ax: free(ax)
        if ay: free(ay)
        if ax_new: free(ax_new)
        if ay_new: free(ay_new)
        raise MemoryError()

    cdef int i, j, step
    cdef double dx, dy, dist2, inv_dist3, fx, fy, ke
    cdef double half_dt2 = 0.5 * dt * dt
    cdef double half_dt = 0.5 * dt

    # Initialize
    for i in range(n):
        px[i] = sin(<double>i)
        py[i] = cos(<double>i)
        vx[i] = 0.0
        vy[i] = 0.0
        ax[i] = 0.0
        ay[i] = 0.0

    # Initial accelerations
    for i in range(n):
        for j in range(i + 1, n):
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            dist2 = dx * dx + dy * dy + soft2
            inv_dist3 = 1.0 / (dist2 * sqrt(dist2))
            fx = dx * inv_dist3
            fy = dy * inv_dist3
            ax[i] += fx
            ay[i] += fy
            ax[j] -= fx
            ay[j] -= fy

    # Verlet loop
    for step in range(steps):
        for i in range(n):
            px[i] += vx[i] * dt + half_dt2 * ax[i]
            py[i] += vy[i] * dt + half_dt2 * ay[i]

        for i in range(n):
            ax_new[i] = 0.0
            ay_new[i] = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                dist2 = dx * dx + dy * dy + soft2
                inv_dist3 = 1.0 / (dist2 * sqrt(dist2))
                fx = dx * inv_dist3
                fy = dy * inv_dist3
                ax_new[i] += fx
                ay_new[i] += fy
                ax_new[j] -= fx
                ay_new[j] -= fy

        for i in range(n):
            vx[i] += half_dt * (ax[i] + ax_new[i])
            vy[i] += half_dt * (ay[i] + ay_new[i])
            ax[i] = ax_new[i]
            ay[i] = ay_new[i]

    ke = 0.0
    for i in range(n):
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])

    free(px)
    free(py)
    free(vx)
    free(vy)
    free(ax)
    free(ay)
    free(ax_new)
    free(ay_new)
    return ke
