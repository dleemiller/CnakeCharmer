"""Simulate batches of projectiles with drag and compute aggregate statistics.

Keywords: physics, projectile, motion, drag, batch simulation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(8000,))
def projectile_motion_batch(n: int) -> tuple:
    """Simulate n projectiles with quadratic drag, compute range and height stats.

    Each projectile i has:
      v0 = 20 + (i % 40), angle = (10 + (i * 7) % 80) degrees
      drag coefficient = 0.005, dt = 0.005

    Simulates until y < 0, records range (final x), max height.

    Args:
        n: Number of projectiles.

    Returns:
        Tuple of (total_range_x100, max_height_x100, final_x_of_first_x100).
        All values multiplied by 100 and truncated to int for exact comparison.
    """
    dt = 0.005
    drag = 0.005
    g = 9.81
    pi = math.pi

    total_range = 0.0
    global_max_height = 0.0
    first_final_x = 0.0

    for i in range(n):
        v0 = 20.0 + (i % 40)
        angle = (10 + (i * 7) % 80) * pi / 180.0
        vx = v0 * math.cos(angle)
        vy = v0 * math.sin(angle)
        x = 0.0
        y = 0.0
        max_h = 0.0

        steps = 0
        while steps < 100000:  # safety limit
            speed = math.sqrt(vx * vx + vy * vy)
            ax = -drag * speed * vx
            ay = -g - drag * speed * vy
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
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

    # Truncate to int after scaling for exact comparison
    return (int(total_range * 100), int(global_max_height * 100), int(first_final_x * 100))
