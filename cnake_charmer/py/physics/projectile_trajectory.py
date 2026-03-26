"""Simulate projectile trajectories with drag and compute max heights.

Keywords: physics, projectile, trajectory, drag, simulation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def projectile_trajectory(n: int) -> float:
    """Simulate n projectile trajectories with air drag.

    v0 = (i%50 + 10), angle = (i%90) * pi/180. dt=0.01, drag=0.01.
    Computes max height for each trajectory, returns sum.

    Args:
        n: Number of projectiles.

    Returns:
        Sum of maximum heights as a float.
    """
    dt = 0.01
    drag = 0.01
    g = 9.81
    pi = math.pi

    total = 0.0
    for i in range(n):
        v0 = (i % 50) + 10.0
        angle = ((i % 90) + 1) * pi / 180.0
        vx = v0 * math.cos(angle)
        vy = v0 * math.sin(angle)
        max_h = 0.0
        y = 0.0

        while vy > 0 or y > 0:
            speed = math.sqrt(vx * vx + vy * vy)
            ax = -drag * speed * vx
            ay = -g - drag * speed * vy
            vx += ax * dt
            vy += ay * dt
            y += vy * dt
            if y < 0:
                break
            if y > max_h:
                max_h = y

        total += max_h

    return total
