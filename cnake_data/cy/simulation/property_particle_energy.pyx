# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate particles with @property getter/setter.

Keywords: simulation, particle, property, setter, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef class Particle:
    """Particle with position property that updates velocity."""
    cdef double _x
    cdef double _y
    cdef double _vx
    cdef double _vy
    cdef double mass

    def __cinit__(
        self, double x, double y, double mass
    ):
        self._x = x
        self._y = y
        self._vx = 0.0
        self._vy = 0.0
        self.mass = mass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, double new_x):
        self._vx = new_x - self._x
        self._x = new_x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, double new_y):
        self._vy = new_y - self._y
        self._y = new_y

    cdef double kinetic_energy(self):
        return 0.5 * self.mass * (
            self._vx * self._vx + self._vy * self._vy
        )


@cython_benchmark(syntax="cy", args=(5000,))
def property_particle_energy(int n):
    """Simulate n particles for 50 steps."""
    cdef list particles = []
    cdef double px, py, mass, dx, dy, total
    cdef int i, step
    cdef long long seed, seed2
    cdef Particle p

    for i in range(n):
        px = (
            (<long long>i * <long long>2654435761 + 17)
            % 10000
        ) / 100.0
        py = (
            (<long long>i * <long long>1103515245 + 12345)
            % 10000
        ) / 100.0
        mass = 1.0 + (i % 5) * 0.5
        particles.append(Particle(px, py, mass))

    for step in range(50):
        for i in range(n):
            p = <Particle>particles[i]
            seed = (
                <long long>i * <long long>1664525
                + <long long>step * <long long>214013
                + <long long>1013904223
            )
            dx = (
                (seed ^ (seed >> 7)) % 1000
            ) / 1000.0 - 0.5
            seed2 = (
                seed * <long long>1103515245 + 12345
            )
            dy = (
                (seed2 ^ (seed2 >> 7)) % 1000
            ) / 1000.0 - 0.5
            p.x = p._x + dx
            p.y = p._y + dy

    total = 0.0
    for i in range(n):
        p = <Particle>particles[i]
        total += p.kinetic_energy()

    return total
