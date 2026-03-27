# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate batches of projectiles with drag (Cython-optimized).

Keywords: physics, projectile, motion, drag, batch simulation, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(8000,))
def projectile_motion_batch(int n):
    """Simulate projectile trajectories with typed variables and libc math."""
    cdef double dt = 0.005
    cdef double drag = 0.005
    cdef double g = 9.81
    cdef double total_range = 0.0
    cdef double global_max_height = 0.0
    cdef double first_final_x = 0.0
    cdef double v0, angle, vx, vy, x, y, max_h, speed, ax, ay
    cdef int i, steps

    for i in range(n):
        v0 = 20.0 + (i % 40)
        angle = (10 + (i * 7) % 80) * M_PI / 180.0
        vx = v0 * cos(angle)
        vy = v0 * sin(angle)
        x = 0.0
        y = 0.0
        max_h = 0.0

        steps = 0
        while steps < 100000:
            speed = sqrt(vx * vx + vy * vy)
            ax = -drag * speed * vx
            ay = -g - drag * speed * vy
            vx = vx + ax * dt
            vy = vy + ay * dt
            x = x + vx * dt
            y = y + vy * dt
            if y < 0:
                break
            if y > max_h:
                max_h = y
            steps += 1

        total_range += x
        if max_h > global_max_height:
            global_max_height = max_h
        if i == 0:
            first_final_x = x

    return (int(total_range * 100), int(global_max_height * 100), int(first_final_x * 100))
