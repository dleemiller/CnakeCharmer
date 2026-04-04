# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Vector3 class operations with cross-product accumulation (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef class Vec3:
    cdef double x
    cdef double y
    cdef double z

    def __cinit__(self, double x, double y, double z):
        self.x = x
        self.y = y
        self.z = z

    cdef inline double norm2(self) noexcept:
        return self.x * self.x + self.y * self.y + self.z * self.z


cdef void _cross_iterate(Vec3 a, Vec3 b, int steps, double *total_out, double *peak_out) noexcept nogil:
    cdef int i
    cdef double cx, cy, cz, n2
    cdef double total = 0.0
    cdef double peak = 0.0

    for i in range(steps):
        cx = a.y * b.z - a.z * b.y
        cy = a.z * b.x - a.x * b.z
        cz = a.x * b.y - a.y * b.x
        n2 = cx * cx + cy * cy + cz * cz
        total += n2 + (i & 1) * 0.01
        if n2 > peak:
            peak = n2
        a.x = cx * 0.9 + 0.1
        a.y = cy * 0.8 - 0.05
        a.z = cz * 0.85 + 0.02

    total_out[0] = total
    peak_out[0] = peak


@cython_benchmark(syntax="cy", args=(450000, 0.13))
def vec3_class_cross_norm(int steps, double bias):
    cdef Vec3 a = Vec3(1.0 + bias, 0.5 - bias, 0.25 + bias)
    cdef Vec3 b = Vec3(0.75, -0.25, 1.25)
    cdef double total = 0.0
    cdef double peak = 0.0

    with nogil:
        _cross_iterate(a, b, steps, &total, &peak)

    return (total, peak, a.norm2())
