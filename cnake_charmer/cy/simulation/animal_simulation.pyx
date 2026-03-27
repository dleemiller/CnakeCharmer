# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate predator and prey animals using cdef class inheritance (Cython).

Keywords: simulation, animals, predator, prey, inheritance, cdef class, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


cdef class Animal:
    """Base animal with position and speed."""
    cdef double x
    cdef double y
    cdef double speed
    cdef double total_distance

    def __cinit__(self, double x, double y, double speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.total_distance = 0.0

    cpdef void move(self, double dx, double dy):
        """Move the animal by (dx, dy) scaled by speed."""
        cdef double adx = dx * self.speed
        cdef double ady = dy * self.speed
        self.x += adx
        self.y += ady
        self.total_distance += sqrt(adx * adx + ady * ady)


cdef class Predator(Animal):
    """Predator moves at 2x base speed."""

    def __cinit__(self, double x, double y, double speed=2.0):
        pass  # Animal.__cinit__ handles initialization

    cpdef void move(self, double dx, double dy):
        """Predators move faster and have a charge bonus on large moves."""
        cdef double mag = sqrt(dx * dx + dy * dy)
        cdef double bonus = 1.5 if mag > 0.5 else 1.0
        cdef double adx = dx * self.speed * bonus
        cdef double ady = dy * self.speed * bonus
        self.x += adx
        self.y += ady
        self.total_distance += sqrt(adx * adx + ady * ady)


cdef class Prey(Animal):
    """Prey moves at 1x base speed."""

    def __cinit__(self, double x, double y, double speed=1.0):
        pass  # Animal.__cinit__ handles initialization

    cpdef void move(self, double dx, double dy):
        """Prey moves cautiously: reduced speed for large moves."""
        cdef double mag = sqrt(dx * dx + dy * dy)
        cdef double dampen = 0.7 if mag > 0.5 else 1.0
        cdef double adx = dx * self.speed * dampen
        cdef double ady = dy * self.speed * dampen
        self.x += adx
        self.y += ady
        self.total_distance += sqrt(adx * adx + ady * ady)


@cython_benchmark(syntax="cy", args=(50000,))
def animal_simulation(int n):
    """Simulate n animals for 20 steps each, return total distance traveled."""
    cdef int k = 20
    cdef list animals = []
    cdef double x, y, dx, dy, total
    cdef int i, step
    cdef long long seed, seed2
    cdef Animal a

    for i in range(n):
        x = ((<long long>i * <long long>2654435761 + 17) % 10000) / 100.0
        y = ((<long long>i * <long long>1103515245 + 12345) % 10000) / 100.0
        if i % 3 == 0:
            animals.append(Predator(x, y, 2.0))
        else:
            animals.append(Prey(x, y, 1.0))

    for step in range(k):
        for i in range(n):
            seed = <long long>i * <long long>1664525 + <long long>step * <long long>214013 + <long long>1013904223
            dx = ((seed ^ (seed >> 7)) % 1000) / 1000.0 - 0.5
            seed2 = seed * <long long>1103515245 + 12345
            dy = ((seed2 ^ (seed2 >> 7)) % 1000) / 1000.0 - 0.5
            a = <Animal>animals[i]
            a.move(dx, dy)

    total = 0.0
    for i in range(n):
        a = <Animal>animals[i]
        total += a.total_distance

    return total
