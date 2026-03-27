"""Simulate particles bouncing in a box and compute final kinetic energy.

Keywords: particle, simulation, bounce, collision, physics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def particle_bounce(n: int) -> float:
    """Simulate n particles bouncing inside a unit box for 200 steps.

    Each particle has position (x, y) and velocity (vx, vy). Particles bounce
    off walls elastically. Returns total kinetic energy (sum of v^2) at the end.

    Args:
        n: Number of particles.

    Returns:
        Total kinetic energy after simulation.
    """
    steps = 200
    dt = 0.01

    # Initialize particles
    px = [0.0] * n
    py = [0.0] * n
    vx = [0.0] * n
    vy = [0.0] * n
    mass = [0.0] * n

    for i in range(n):
        h = (i * 2654435761) & 0xFFFFFFFF
        px[i] = (h % 1000) / 1000.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        py[i] = (h % 1000) / 1000.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        vx[i] = ((h % 2001) - 1000) / 100.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        vy[i] = ((h % 2001) - 1000) / 100.0
        mass[i] = 1.0 + (i % 5) * 0.5

    for _ in range(steps):
        for i in range(n):
            px[i] += vx[i] * dt
            py[i] += vy[i] * dt

            # Bounce off walls [0, 1]
            if px[i] < 0.0:
                px[i] = -px[i]
                vx[i] = -vx[i]
            elif px[i] > 1.0:
                px[i] = 2.0 - px[i]
                vx[i] = -vx[i]

            if py[i] < 0.0:
                py[i] = -py[i]
                vy[i] = -vy[i]
            elif py[i] > 1.0:
                py[i] = 2.0 - py[i]
                vy[i] = -vy[i]

    total_ke = 0.0
    for i in range(n):
        total_ke += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i])

    return total_ke
