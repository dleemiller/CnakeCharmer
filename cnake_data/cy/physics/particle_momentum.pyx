# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Particle momentum computation with extension types (Cython-optimized).

Keywords: particle, momentum, kinetic_energy, physics, simulation, class, cython
"""

from cnake_data.benchmarks import cython_benchmark
from libc.math cimport sqrt


cdef class Particle:
    """Simple particle with mass, position, and velocity."""
    cdef public double mass
    cdef public double position
    cdef double velocity

    def __init__(self, double mass, double position, double velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

    cpdef double get_momentum(self):
        return self.mass * self.velocity

    cpdef double get_energy(self):
        return self.mass * self.velocity * self.velocity / 2.0

    cpdef void set_energy(self, double e):
        self.velocity = sqrt(e * 2.0 / self.mass)


@cython_benchmark(syntax="cy", args=(200000,))
def particle_momentum_sum(int n):
    """Create n particles and sum their momenta."""
    cdef double total = 0.0
    cdef double mass, velocity
    cdef int i
    cdef Particle p
    for i in range(n):
        mass = 1.0 + (i % 100) * 0.1
        velocity = 2.0 + (i % 50) * 0.05
        p = Particle(mass, 0.0, velocity)
        total += p.get_momentum()
    return total
