"""Simulate n particles with position, velocity, and mass.

Demonstrates nested struct pattern: Vec2 inside Particle.

Keywords: simulation, particle, nested struct, physics, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def nested_struct_particle(n: int) -> float:
    """Simulate n particles for 50 timesteps.

    Each particle has pos (x,y), vel (x,y), and mass.
    Returns sum of final distances from origin.

    Args:
        n: Number of particles.

    Returns:
        Sum of distances from origin after simulation.
    """
    steps = 50
    dt = 0.01

    # Initialize particles
    px = [0.0] * n
    py_ = [0.0] * n
    vx = [0.0] * n
    vy = [0.0] * n
    mass = [0.0] * n

    for i in range(n):
        h = ((i * 2654435761) ^ (i * 2246822519)) & 0xFFFFFFFF
        px[i] = (h & 0xFFFF) / 65535.0 * 10.0 - 5.0
        py_[i] = ((h >> 16) & 0xFFFF) / 65535.0 * 10.0 - 5.0
        h2 = ((i * 1664525 + 1013904223) ^ (i * 214013)) & 0xFFFFFFFF
        vx[i] = (h2 & 0xFFFF) / 65535.0 * 2.0 - 1.0
        vy[i] = ((h2 >> 16) & 0xFFFF) / 65535.0 * 2.0 - 1.0
        mass[i] = 0.5 + (i % 10) * 0.1

    # Simple gravity toward origin simulation
    for _s in range(steps):
        for i in range(n):
            dist_sq = px[i] * px[i] + py_[i] * py_[i] + 0.01
            force = -mass[i] / dist_sq
            vx[i] += force * px[i] * dt
            vy[i] += force * py_[i] * dt
            px[i] += vx[i] * dt
            py_[i] += vy[i] * dt

    total_dist = 0.0
    for i in range(n):
        total_dist += (px[i] * px[i] + py_[i] * py_[i]) ** 0.5
    return total_dist
