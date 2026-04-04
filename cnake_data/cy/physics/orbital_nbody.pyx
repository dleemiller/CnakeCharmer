# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate 3-body orbital system using Verlet integration (Cython-optimized).

Keywords: physics, orbital, nbody, gravity, simulation, verlet, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def orbital_nbody(int n):
    """Simulate a 3-body gravitational system for n timesteps."""
    cdef double G = 1.0
    cdef double dt = 1e-4
    cdef double softening = 1e-6
    cdef int nb = 3
    cdef int i, j, step

    cdef double mass[3]
    cdef double px[3]
    cdef double py_[3]
    cdef double vx[3]
    cdef double vy[3]
    cdef double ax[3]
    cdef double ay[3]
    cdef double ax_new[3]
    cdef double ay_new[3]

    mass[0] = 1.0; mass[1] = 0.5; mass[2] = 0.3
    px[0] = 0.0; px[1] = 1.0; px[2] = -0.5
    py_[0] = 0.0; py_[1] = 0.0; py_[2] = 0.866
    vx[0] = 0.0; vx[1] = 0.0; vx[2] = 0.0
    vy[0] = 0.1; vy[1] = -0.2; vy[2] = 0.1

    cdef double dx, dy, r2, r, r3, f, ke, pe, total_energy

    # Initial accelerations
    for i in range(nb):
        ax[i] = 0.0
        ay[i] = 0.0
        for j in range(nb):
            if i == j:
                continue
            dx = px[j] - px[i]
            dy = py_[j] - py_[i]
            r2 = dx * dx + dy * dy + softening
            r = sqrt(r2)
            r3 = r2 * r
            f = G * mass[j] / r3
            ax[i] = ax[i] + f * dx
            ay[i] = ay[i] + f * dy

    for step in range(n):
        for i in range(nb):
            px[i] = px[i] + vx[i] * dt + 0.5 * ax[i] * dt * dt
            py_[i] = py_[i] + vy[i] * dt + 0.5 * ay[i] * dt * dt

        for i in range(nb):
            ax_new[i] = 0.0
            ay_new[i] = 0.0
            for j in range(nb):
                if i == j:
                    continue
                dx = px[j] - px[i]
                dy = py_[j] - py_[i]
                r2 = dx * dx + dy * dy + softening
                r = sqrt(r2)
                r3 = r2 * r
                f = G * mass[j] / r3
                ax_new[i] = ax_new[i] + f * dx
                ay_new[i] = ay_new[i] + f * dy

        for i in range(nb):
            vx[i] = vx[i] + 0.5 * (ax[i] + ax_new[i]) * dt
            vy[i] = vy[i] + 0.5 * (ay[i] + ay_new[i]) * dt
            ax[i] = ax_new[i]
            ay[i] = ay_new[i]

    ke = 0.0
    for i in range(nb):
        ke = ke + 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i])

    pe = 0.0
    for i in range(nb):
        for j in range(i + 1, nb):
            dx = px[j] - px[i]
            dy = py_[j] - py_[i]
            r = sqrt(dx * dx + dy * dy + softening)
            pe = pe - G * mass[i] * mass[j] / r

    total_energy = ke + pe
    return (px[0], py_[0], total_energy)
