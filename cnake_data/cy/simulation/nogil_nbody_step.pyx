# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""N-body gravity simulation with GIL release.

Keywords: simulation, n-body, gravity, nogil, physics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt
from cnake_data.benchmarks import cython_benchmark


cdef void _gravity_step(
    double *px, double *py,
    double *vx, double *vy,
    int n, double dt, double soft2
) noexcept nogil:
    """Compute one gravity step: forces, update vel, pos."""
    cdef double *ax = <double *>malloc(
        n * sizeof(double)
    )
    cdef double *ay = <double *>malloc(
        n * sizeof(double)
    )
    if not ax or not ay:
        if ax:
            free(ax)
        if ay:
            free(ay)
        return

    cdef int i, j
    cdef double dx, dy, dist2, inv_dist3, fx, fy

    for i in range(n):
        ax[i] = 0.0
        ay[i] = 0.0

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

    for i in range(n):
        vx[i] += ax[i] * dt
        vy[i] += ay[i] * dt
        px[i] += vx[i] * dt
        py[i] += vy[i] * dt

    free(ax)
    free(ay)


@cython_benchmark(syntax="cy", args=(500,))
def nogil_nbody_step(int n):
    """Simulate n particles under gravity for 10 steps."""
    cdef double dt = 0.0005
    cdef double soft2 = 0.0001
    cdef int steps = 10

    cdef double *px = <double *>malloc(
        n * sizeof(double)
    )
    cdef double *py = <double *>malloc(
        n * sizeof(double)
    )
    cdef double *vx = <double *>malloc(
        n * sizeof(double)
    )
    cdef double *vy = <double *>malloc(
        n * sizeof(double)
    )
    if not px or not py or not vx or not vy:
        if px:
            free(px)
        if py:
            free(py)
        if vx:
            free(vx)
        if vy:
            free(vy)
        raise MemoryError()

    cdef int i, step
    cdef double ke

    for i in range(n):
        px[i] = sin(<double>i * 0.1)
        py[i] = cos(<double>i * 0.1)
        vx[i] = 0.0
        vy[i] = 0.0

    with nogil:
        for step in range(steps):
            _gravity_step(
                px, py, vx, vy, n, dt, soft2
            )

    ke = 0.0
    for i in range(n):
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])

    free(px)
    free(py)
    free(vx)
    free(vy)
    return ke
