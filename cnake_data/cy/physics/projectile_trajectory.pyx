# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate projectile trajectories with drag (Cython-optimized).

Keywords: physics, projectile, trajectory, drag, simulation, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def projectile_trajectory(int n):
    """Simulate projectile trajectories with typed variables and libc math."""
    cdef double dt = 0.01
    cdef double drag = 0.01
    cdef double g = 9.81
    cdef double total = 0.0
    cdef double v0, angle, vx, vy, max_h, y, speed, ax, ay
    cdef int i

    for i in range(n):
        v0 = (i % 50) + 10.0
        angle = ((i % 90) + 1) * M_PI / 180.0
        vx = v0 * cos(angle)
        vy = v0 * sin(angle)
        max_h = 0.0
        y = 0.0

        while vy > 0 or y > 0:
            speed = sqrt(vx * vx + vy * vy)
            ax = -drag * speed * vx
            ay = -g - drag * speed * vy
            vx = vx + ax * dt
            vy = vy + ay * dt
            y = y + vy * dt
            if y < 0:
                break
            if y > max_h:
                max_h = y

        total += max_h

    return total
