# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count points inside a triangle using barycentric coordinates (Cython-optimized).

Keywords: geometry, point in triangle, barycentric, classification, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def point_in_triangle(int n):
    """Count how many of n test points lie inside a fixed triangle."""
    cdef double x1 = 0.0, y1 = 0.0
    cdef double x2 = 100.0, y2 = 0.0
    cdef double x3 = 50.0, y3 = 86.0
    cdef double denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    cdef int count = 0
    cdef int i
    cdef double px, py, a, b, c

    for i in range(n):
        px = <double>((i * 17 + 3) % 200 - 50)
        py = <double>((i * 13 + 7) % 200 - 50)

        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        c = 1.0 - a - b

        if a >= 0.0 and b >= 0.0 and c >= 0.0:
            count += 1

    return count
