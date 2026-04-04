"""Simulate particles with property getter/setter for position.

Keywords: simulation, particle, property, setter, energy, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Particle:
    """Particle with position property that updates velocity."""

    def __init__(self, x, y, mass):
        self._x = x
        self._y = y
        self._vx = 0.0
        self._vy = 0.0
        self.mass = mass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_x):
        self._vx = new_x - self._x
        self._x = new_x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_y):
        self._vy = new_y - self._y
        self._y = new_y

    def kinetic_energy(self):
        return 0.5 * self.mass * (self._vx**2 + self._vy**2)


@python_benchmark(args=(5000,))
def property_particle_energy(n: int) -> float:
    """Simulate n particles for 50 steps, return total energy.

    Args:
        n: Number of particles.

    Returns:
        Total kinetic energy of all particles.
    """
    particles = []
    for i in range(n):
        px = ((i * 2654435761 + 17) % 10000) / 100.0
        py = ((i * 1103515245 + 12345) % 10000) / 100.0
        mass = 1.0 + (i % 5) * 0.5
        particles.append(Particle(px, py, mass))

    for step in range(50):
        for i in range(n):
            p = particles[i]
            seed = i * 1664525 + step * 214013 + 1013904223
            dx = ((seed ^ (seed >> 7)) % 1000) / 1000.0 - 0.5
            seed2 = seed * 1103515245 + 12345
            dy = ((seed2 ^ (seed2 >> 7)) % 1000) / 1000.0 - 0.5
            p.x = p.x + dx
            p.y = p.y + dy

    total = 0.0
    for p in particles:
        total += p.kinetic_energy()

    return total
