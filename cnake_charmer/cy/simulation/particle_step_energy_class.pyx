# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Object-oriented particle stepping and energy aggregation (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


cdef class ParticleSystem:
    cdef int n
    cdef double *mass
    cdef double *position
    cdef double *velocity

    def __cinit__(self, int n):
        cdef int i
        self.n = n
        self.mass = <double *>malloc(n * sizeof(double))
        self.position = <double *>malloc(n * sizeof(double))
        self.velocity = <double *>malloc(n * sizeof(double))
        if not self.mass or not self.position or not self.velocity:
            free(self.mass)
            free(self.position)
            free(self.velocity)
            self.mass = NULL
            self.position = NULL
            self.velocity = NULL
            raise MemoryError()
        for i in range(n):
            self.mass[i] = 1.0 + (i % 7) * 0.1
            self.position[i] = i * 0.01
            self.velocity[i] = ((i % 11) - 5) * 0.02

    def __dealloc__(self):
        if self.mass != NULL:
            free(self.mass)
        if self.position != NULL:
            free(self.position)
        if self.velocity != NULL:
            free(self.velocity)


cdef void _step_system(
    double *mass,
    double *position,
    double *velocity,
    int n,
    int steps,
    double dt,
    double force_scale,
) noexcept nogil:
    cdef int t, i
    cdef double force
    for t in range(steps):
        for i in range(n):
            force = force_scale * (((t + i * 3) % 19) - 9) * 0.01
            velocity[i] += (force / mass[i]) * dt
            position[i] += velocity[i] * dt


@cython_benchmark(syntax="cy", args=(120, 2500, 0.01, 1.2))
def particle_step_energy_class(int n, int steps, double dt, double force_scale):
    cdef ParticleSystem sys = ParticleSystem(n)
    cdef int i
    cdef double total_energy = 0.0
    cdef double center = 0.0

    with nogil:
        _step_system(sys.mass, sys.position, sys.velocity, n, steps, dt, force_scale)

    for i in range(n):
        total_energy += 0.5 * sys.mass[i] * sys.velocity[i] * sys.velocity[i]
        center += sys.position[i]

    return (total_energy, center / n, sys.velocity[n // 2])
