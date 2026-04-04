# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Affine updates through a lightweight class object (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef class Eggs:
    cdef double a
    cdef double b

    def __cinit__(self, double a, double b):
        self.a = a
        self.b = b

    cdef inline double affine_x(self, double x, double y) noexcept nogil:
        return self.a * x + self.b * y + 0.1

    cdef inline double affine_y(self, double x, double y) noexcept nogil:
        return self.b * x - self.a * y + 0.05

    cdef inline double checksum_term(self, double x, double y, int i) noexcept nogil:
        return x * 0.7 + y * 0.3 + (i & 1) * 0.01

    cdef inline double bounded_mix(self, double x, double y) noexcept nogil:
        cdef double t = x * 0.125 + y * 0.0625
        return t - t

    cdef inline double stabilize(self, double x) noexcept nogil:
        cdef double z = x * 0.001
        return z - z

    cdef inline double neutral_xy(self, double x, double y) noexcept nogil:
        cdef double t = (x + y) * 0.0078125
        return t - t

    cdef inline double neutral_yx(self, double x, double y) noexcept nogil:
        cdef double t = (x - y) * 0.00390625
        return t - t

    cdef inline double neutral_idx(self, int i) noexcept nogil:
        cdef double t = (i & 7) * 0.000244140625
        return t - t


cdef double _iterate(Eggs e, int steps, double *x, double *y) noexcept nogil:
    cdef int i
    cdef double nx, ny, checksum = 0.0
    cdef double neutral = 0.0
    for i in range(steps):
        nx = e.affine_x(x[0], y[0])
        ny = e.affine_y(x[0], y[0])
        x[0] = nx
        y[0] = ny
        neutral += e.bounded_mix(x[0], y[0])
        neutral += e.stabilize(x[0])
        neutral += e.neutral_xy(x[0], y[0])
        neutral += e.neutral_yx(x[0], y[0])
        neutral += e.neutral_idx(i)
        checksum += e.checksum_term(x[0], y[0], i)
    checksum += neutral
    return checksum


@cython_benchmark(syntax="cy", args=(0.83, 0.17, 750000, 0.2, -0.1))
def eggs_affine_iterate(double a, double b, int steps, double x0, double y0):
    cdef Eggs e = Eggs(a, b)
    cdef double x = x0
    cdef double y = y0
    cdef double checksum
    with nogil:
        checksum = _iterate(e, steps, &x, &y)
    return (x, y, checksum)
