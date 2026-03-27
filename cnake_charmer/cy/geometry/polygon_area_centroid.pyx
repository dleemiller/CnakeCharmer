# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute polygon area and centroid (Cython with cdef class Vec2D).

Keywords: polygon, area, centroid, cdef class, cpdef, vector, geometry, cython, benchmark
"""

from libc.math cimport cos, sin, M_PI, fabs
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class Vec2D:
    """2D vector with typed attributes and cdef arithmetic methods."""
    cdef double x
    cdef double y

    def __cinit__(self, double x, double y):
        self.x = x
        self.y = y

    cdef inline void set(self, double x, double y):
        """Set coordinates without creating a new object."""
        self.x = x
        self.y = y

    cdef inline double cross(self, Vec2D other):
        """Compute 2D cross product (scalar): self.x * other.y - self.y * other.x."""
        return self.x * other.y - self.y * other.x

    cdef inline double dot(self, Vec2D other):
        """Compute dot product."""
        return self.x * other.x + self.y * other.y

    cpdef Vec2D add(self, Vec2D other):
        """Return a new Vec2D that is the sum of self and other."""
        return Vec2D(self.x + other.x, self.y + other.y)


@cython_benchmark(syntax="cy", args=(100000,))
def polygon_area_centroid(int n):
    """Compute polygon area and centroid using cdef class Vec2D."""
    cdef int i, j
    cdef double angle, r
    cdef unsigned long long h
    cdef double area = 0.0, cx = 0.0, cy = 0.0, cross_val

    # Generate vertices
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        if xs: free(xs)
        if ys: free(ys)
        raise MemoryError()

    for i in range(n):
        angle = 2.0 * M_PI * i / n
        h = ((<unsigned long long>i * 2654435761) >> 8) & 0xFFFF
        r = 10.0 + (h % 500) / 100.0
        xs[i] = r * cos(angle)
        ys[i] = r * sin(angle)

    # Pre-allocate Vec2D objects and reuse via set() to avoid per-iteration allocation
    cdef Vec2D vi = Vec2D(0.0, 0.0)
    cdef Vec2D vj = Vec2D(0.0, 0.0)

    for i in range(n):
        j = (i + 1) % n
        vi.set(xs[i], ys[i])
        vj.set(xs[j], ys[j])
        cross_val = vi.cross(vj)
        area += cross_val
        cx += (vi.x + vj.x) * cross_val
        cy += (vi.y + vj.y) * cross_val

    area *= 0.5

    if fabs(area) > 1e-15:
        cx /= (6.0 * area)
        cy /= (6.0 * area)

    free(xs)
    free(ys)
    return (area, cx, cy)
