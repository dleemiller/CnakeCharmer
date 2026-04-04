# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find diameter of convex hull using rotating calipers (Cython-optimized).

Keywords: geometry, convex hull, rotating calipers, diameter, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sqrt, sin, cos
from cnake_data.benchmarks import cython_benchmark


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
def rotating_calipers(int n):
    """Find the diameter of the convex hull of n deterministic 2D points.

    Uses C arrays, qsort, Andrew's monotone chain, and rotating calipers.

    Args:
        n: Number of points.

    Returns:
        The diameter of the convex hull.
    """
    cdef int i, j, ni, nj, h, m, top
    cdef double dx, dy, dist_sq, max_dist_sq, eix, eiy, ejx, ejy, cr

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
        return 0.0
    if m == 2:
        dx = pts[1].x - pts[0].x
        dy = pts[1].y - pts[0].y
        free(pts)
        return sqrt(dx * dx + dy * dy)

    # Andrew's monotone chain - hull can be at most 2*m points
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
    cdef int lower_size = top + 1  # +1 for separator
    for i in range(m - 2, -1, -1):
        while top >= lower_size and cross2d(hull[top - 2].x, hull[top - 2].y,
                                             hull[top - 1].x, hull[top - 1].y,
                                             pts[i].x, pts[i].y) <= 0:
            top -= 1
        hull[top] = pts[i]
        top += 1

    # Remove last point (duplicate of first)
    h = top - 1

    free(pts)

    if h <= 1:
        free(hull)
        return 0.0
    if h == 2:
        dx = hull[1].x - hull[0].x
        dy = hull[1].y - hull[0].y
        free(hull)
        return sqrt(dx * dx + dy * dy)

    # Rotating calipers
    max_dist_sq = 0.0
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        while True:
            nj = (j + 1) % h
            eix = hull[ni].x - hull[i].x
            eiy = hull[ni].y - hull[i].y
            ejx = hull[nj].x - hull[j].x
            ejy = hull[nj].y - hull[j].y
            cr = eix * ejy - eiy * ejx
            if cr > 0:
                j = nj
            else:
                break

        dx = hull[i].x - hull[j].x
        dy = hull[i].y - hull[j].y
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq

    free(hull)
    return sqrt(max_dist_sq)
