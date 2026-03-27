# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Chain affine transforms with not-None enforcement.

Keywords: numerical, transform, not none, extension type, affine, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef class Transform:
    """Affine transform: y = scale * x + offset."""
    cdef double scale
    cdef double offset

    def __cinit__(self, double scale, double offset):
        self.scale = scale
        self.offset = offset

    def apply(self, Transform other not None):
        """Compose self(other(x))."""
        return Transform(
            self.scale * other.scale,
            self.scale * other.offset + self.offset,
        )


@cython_benchmark(syntax="cy", args=(50000,))
def not_none_transform(int n):
    """Chain n transforms, return final scale + offset."""
    cdef Transform result = Transform(1.0, 0.0)
    cdef int i
    cdef double s, o, norm
    cdef Transform t

    for i in range(n):
        s = ((<long long>i * <long long>2654435761)
             % 1000) / 500.0
        o = ((<long long>i * <long long>1664525
              + <long long>1013904223) % 1000) / 500.0
        t = Transform(s, o)
        result = result.apply(t)
        norm = (fabs(result.scale)
                + fabs(result.offset) + 1.0)
        result = Transform(
            result.scale / norm, result.offset / norm
        )
    return result.scale + result.offset
