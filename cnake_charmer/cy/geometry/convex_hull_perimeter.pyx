# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the perimeter of the convex hull of n 2D points (Cython-optimized).

Keywords: convex hull, perimeter, geometry, monotone chain, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sqrt, sin, cos
from cnake_charmer.benchmarks import cython_benchmark


cdef struct Point:
    double x
    double y


cdef int point_cmp(const void *a, const void *b) noexcept nogil:
    cdef Point *pa = <Point *>a
    cdef Point *pb = <Point *>b
    if pa.x < pb.x:
        return -1
    elif pa.x > pb.x:
        return 1
    elif pa.y < pb.y:
        return -1
    elif pa.y > pb.y:
        return 1
    return 0


cdef inline double cross2d(double ox, double oy, double ax, double ay, double bx, double by) noexcept nogil:
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


@cython_benchmark(syntax="cy", args=(50000,))
def convex_hull_perimeter(int n):
    """Compute the perimeter and vertex count of the convex hull of n deterministic points."""
    cdef int i, j, h, m, top
    cdef double dx, dy, perimeter

    cdef Point *pts = <Point *>malloc(n * sizeof(Point))
    if pts == NULL:
        raise MemoryError()

    # Generate points
    for i in range(n):
        pts[i].x = sin(i * 0.7) * 100.0
        pts[i].y = cos(i * 1.3) * 100.0

    # Sort
    qsort(pts, n, sizeof(Point), point_cmp)

    # Remove duplicates
    m = 1
    for i in range(1, n):
        if pts[i].x != pts[m - 1].x or pts[i].y != pts[m - 1].y:
            pts[m] = pts[i]
            m += 1

    if m == 1:
        free(pts)
        return (0.0, 1)
    if m == 2:
        dx = pts[1].x - pts[0].x
        dy = pts[1].y - pts[0].y
        free(pts)
        return (2.0 * sqrt(dx * dx + dy * dy), 2)

    # Andrew's monotone chain
    cdef Point *hull = <Point *>malloc(2 * m * sizeof(Point))
    if hull == NULL:
        free(pts)
        raise MemoryError()

    # Lower hull
    top = 0
    for i in range(m):
        while top >= 2 and cross2d(hull[top - 2].x, hull[top - 2].y,
                                    hull[top - 1].x, hull[top - 1].y,
                                    pts[i].x, pts[i].y) <= 0:
            top -= 1
        hull[top] = pts[i]
        top += 1

    # Upper hull
    cdef int lower_size = top + 1
    for i in range(m - 2, -1, -1):
        while top >= lower_size and cross2d(hull[top - 2].x, hull[top - 2].y,
                                             hull[top - 1].x, hull[top - 1].y,
                                             pts[i].x, pts[i].y) <= 0:
            top -= 1
        hull[top] = pts[i]
        top += 1

    h = top - 1  # remove duplicate last point

    free(pts)

    # Compute perimeter
    perimeter = 0.0
    for i in range(h):
        j = (i + 1) % h
        dx = hull[j].x - hull[i].x
        dy = hull[j].y - hull[i].y
        perimeter += sqrt(dx * dx + dy * dy)

    free(hull)
    return (perimeter, h)
