"""Object-oriented particle stepping and energy aggregation.

Keywords: simulation, class, particle, integration, energy, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Particle:
    def __init__(self, mass: float, position: float, velocity: float):
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def step(self, force: float, dt: float) -> None:
        self.velocity += (force / self.mass) * dt
        self.position += self.velocity * dt


@python_benchmark(args=(120, 2500, 0.01, 1.2))
def particle_step_energy_class(n: int, steps: int, dt: float, force_scale: float) -> tuple:
    particles = [Particle(1.0 + (i % 7) * 0.1, i * 0.01, ((i % 11) - 5) * 0.02) for i in range(n)]
    for t in range(steps):
        for i, p in enumerate(particles):
            force = force_scale * (((t + i * 3) % 19) - 9) * 0.01
            p.step(force, dt)
    total_energy = 0.0
    center = 0.0
    for p in particles:
        total_energy += 0.5 * p.mass * p.velocity * p.velocity
        center += p.position
    return (total_energy, center / n, particles[n // 2].velocity)
