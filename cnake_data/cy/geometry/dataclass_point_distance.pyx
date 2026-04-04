# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of pairwise distances for 3D points.

Keywords: geometry, point, 3d, distance, dataclass, extension type, cython, benchmark
"""

cimport cython
import cython.dataclasses
from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython.dataclasses.dataclass
cdef class Point3D:
    cdef public double x
    cdef public double y
    cdef public double z


@cython_benchmark(syntax="cy", args=(5000,))
def dataclass_point_distance(int n):
    """Create n 3D points, sum consecutive distances."""
    cdef list points = [None] * n
    cdef int i
    cdef double x, y, z

    for i in range(n):
        x = ((<long long>i * <long long>2654435761)
             % 10000) / 100.0
        y = ((<long long>i * <long long>1664525
              + <long long>1013904223) % 10000) / 100.0
        z = ((<long long>i * <long long>1103515245
              + 12345) % 10000) / 100.0
        points[i] = Point3D(x=x, y=y, z=z)

    cdef double total = 0.0
    cdef double dx, dy, dz
    cdef Point3D pa, pb
    for i in range(n - 1):
        pa = <Point3D>points[i]
        pb = <Point3D>points[i + 1]
        dx = pb.x - pa.x
        dy = pb.y - pa.y
        dz = pb.z - pa.z
        total += sqrt(dx * dx + dy * dy + dz * dz)
    return total
