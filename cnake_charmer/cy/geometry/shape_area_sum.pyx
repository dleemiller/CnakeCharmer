# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum areas of mixed shapes using cdef class inheritance (Cython).

Keywords: geometry, shapes, area, circle, rectangle, inheritance, cdef class, cython, benchmark
"""

from libc.math cimport M_PI
from cnake_charmer.benchmarks import cython_benchmark


cdef class Shape:
    """Base shape with cpdef area method."""

    cpdef double area(self):
        return 0.0


cdef class Circle(Shape):
    """Circle shape with radius."""
    cdef double radius

    def __cinit__(self, double radius):
        self.radius = radius

    cpdef double area(self):
        return M_PI * self.radius * self.radius


cdef class Rectangle(Shape):
    """Rectangle shape with width and height."""
    cdef double width
    cdef double height

    def __cinit__(self, double width, double height):
        self.width = width
        self.height = height

    cpdef double area(self):
        return self.width * self.height


@cython_benchmark(syntax="cy", args=(200000,))
def shape_area_sum(int n):
    """Create n shapes (circles and rectangles), sum their areas."""
    cdef list shapes = []
    cdef int i, kind
    cdef double radius, width, height, total
    cdef Shape s

    for i in range(n):
        kind = ((<long long>i * <long long>2654435761) >> 4) & 1
        if kind == 0:
            radius = ((<long long>i * <long long>1664525 + <long long>1013904223) % 10000) / 100.0
            shapes.append(Circle(radius))
        else:
            width = ((<long long>i * <long long>1103515245 + 12345) % 10000) / 100.0
            height = ((<long long>i * <long long>214013 + <long long>2531011) % 10000) / 100.0
            shapes.append(Rectangle(width, height))

    total = 0.0
    for i in range(n):
        s = <Shape>shapes[i]
        total += s.area()

    return total
