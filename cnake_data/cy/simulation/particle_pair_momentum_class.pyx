# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pairwise particle momentum exchange using class objects (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef class ParticlePairs:
    cdef int n
    cdef double *mass
    cdef double *vx
    cdef double *vy

    def __cinit__(self, int n):
        cdef int i
        self.n = n
        self.mass = <double *>malloc(n * sizeof(double))
        self.vx = <double *>malloc(n * sizeof(double))
        self.vy = <double *>malloc(n * sizeof(double))
        if not self.mass or not self.vx or not self.vy:
            free(self.mass)
            free(self.vx)
            free(self.vy)
            self.mass = NULL
            self.vx = NULL
            self.vy = NULL
            raise MemoryError()
        for i in range(n):
            self.mass[i] = 1.0 + (i % 5) * 0.2
            self.vx[i] = (i % 9) * 0.03
            self.vy[i] = (i % 11) * -0.02

    def __dealloc__(self):
        if self.mass != NULL:
            free(self.mass)
        if self.vx != NULL:
            free(self.vx)
        if self.vy != NULL:
            free(self.vy)


cdef void _exchange_pairs(double *mass, double *vx, double *vy, int n, int rounds, double coupling) noexcept nogil:
    cdef int r, i
    cdef double k, dvx, dvy
    for r in range(rounds):
        for i in range(0, n - 1, 2):
            k = coupling + (r & 3) * 0.005
            dvx = (vx[i + 1] - vx[i]) * k
            dvy = (vy[i + 1] - vy[i]) * k
            vx[i] += dvx / mass[i]
            vy[i] += dvy / mass[i]
            vx[i + 1] -= dvx / mass[i + 1]
            vy[i + 1] -= dvy / mass[i + 1]


@cython_benchmark(syntax="cy", args=(90, 350, 0.07))
def particle_pair_momentum_class(int n, int rounds, double coupling):
    cdef ParticlePairs ps = ParticlePairs(n)
    cdef int i
    cdef double px = 0.0
    cdef double py = 0.0

    with nogil:
        _exchange_pairs(ps.mass, ps.vx, ps.vy, n, rounds, coupling)

    for i in range(n):
        px += ps.mass[i] * ps.vx[i]
        py += ps.mass[i] * ps.vy[i]

    return (px, py, ps.vx[n // 3])
