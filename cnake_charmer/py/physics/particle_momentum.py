"""Particle momentum computation with extension types.

Models particles with mass, position, and velocity, computing momentum
and kinetic energy. Demonstrates Cython extension type optimization.

Keywords: particle, momentum, kinetic_energy, physics, simulation, class
"""

import math

from cnake_charmer.benchmarks import python_benchmark


class Particle:
    """Simple particle with mass, position, and velocity."""

    def __init__(self, mass: float, position: float, velocity: float):
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def get_momentum(self) -> float:
        return self.mass * self.velocity

    def get_energy(self) -> float:
        return self.mass * self.velocity * self.velocity / 2.0

    def set_energy(self, e: float):
        self.velocity = math.sqrt(e * 2.0 / self.mass)


@python_benchmark(args=(200000,))
def particle_momentum_sum(n: int) -> float:
    """Create n particles and sum their momenta.

    Args:
        n: Number of particles to create.

    Returns:
        Sum of all particle momenta.
    """
    total = 0.0
    for i in range(n):
        mass = 1.0 + (i % 100) * 0.1
        velocity = 2.0 + (i % 50) * 0.05
        p = Particle(mass, 0.0, velocity)
        total += p.get_momentum()
    return total
