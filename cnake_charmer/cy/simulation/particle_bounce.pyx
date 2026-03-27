# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate particles bouncing in a box (Cython with cdef class and @property).

Keywords: particle, simulation, cdef class, property, bounce, physics, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class Particle:
    """A particle with position, velocity, and mass using typed C attributes."""
    cdef public double x, y, vx, vy, mass

    def __cinit__(self, double x, double y, double vx, double vy, double mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass

    @property
    def kinetic_energy(self):
        """Compute kinetic energy: 0.5 * mass * v^2."""
        return 0.5 * self.mass * (self.vx * self.vx + self.vy * self.vy)

    @property
    def speed_squared(self):
        """Compute speed squared: vx^2 + vy^2."""
        return self.vx * self.vx + self.vy * self.vy

    cdef void step(self, double dt):
        """Advance the particle by dt and bounce off [0,1] walls."""
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x < 0.0:
            self.x = -self.x
            self.vx = -self.vx
        elif self.x > 1.0:
            self.x = 2.0 - self.x
            self.vx = -self.vx

        if self.y < 0.0:
            self.y = -self.y
            self.vy = -self.vy
        elif self.y > 1.0:
            self.y = 2.0 - self.y
            self.vy = -self.vy


@cython_benchmark(syntax="cy", args=(5000,))
def particle_bounce(int n):
    """Simulate n particles bouncing in a box using cdef class with @property."""
    cdef int steps = 200
    cdef double dt = 0.01
    cdef int i, s
    cdef double total_ke = 0.0
    cdef unsigned long long h
    cdef double px, py, pvx, pvy, m

    # Create particle list
    particles = []
    for i in range(n):
        h = (<unsigned long long>i * 2654435761) & 0xFFFFFFFF
        px = (h % 1000) / 1000.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        py = (h % 1000) / 1000.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        pvx = ((<long long>(h % 2001)) - 1000) / 100.0
        h = (h * 6364136223846793005 + 1) & 0xFFFFFFFF
        pvy = ((<long long>(h % 2001)) - 1000) / 100.0
        m = 1.0 + (i % 5) * 0.5
        particles.append(Particle(px, py, pvx, pvy, m))

    cdef Particle p
    for s in range(steps):
        for i in range(n):
            p = <Particle>particles[i]
            p.step(dt)

    for i in range(n):
        p = <Particle>particles[i]
        total_ke += p.kinetic_energy

    return total_ke
