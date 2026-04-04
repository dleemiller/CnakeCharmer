# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Rasterize rotated ellipses and rectangles on an integer grid (Cython)."""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF

cdef inline double sin_t(double theta) noexcept nogil:
    cdef double x = theta
    cdef double x2 = x * x
    return x - x * x2 / 6.0 + x * x2 * x2 / 120.0

cdef inline double cos_t(double theta) noexcept nogil:
    cdef double x2 = theta * theta
    return 1.0 - x2 / 2.0 + x2 * x2 / 24.0


@cython_benchmark(syntax="cy", args=(48, 36, 48, 17))
def raster_shape_coverage(int width, int height, int n_shapes, int seed):
    cdef int size = width * height
    cdef int *grid = <int *>malloc(size * sizeof(int))
    cdef unsigned int state = <unsigned int>((seed * 1103515245 + 12345) & MASK32)
    cdef int s, x, y, i, cx, cy, sx, sy, covered=0, overlap=0
    cdef unsigned int checksum = 0
    cdef double ang, ca, sa, rx, ry
    cdef bint hit

    if grid == NULL:
        raise MemoryError()
    for i in range(size):
        grid[i] = 0

    for s in range(n_shapes):
        state = (state * 1103515245 + 12345) & MASK32; cx = state % width
        state = (state * 1103515245 + 12345) & MASK32; cy = state % height
        state = (state * 1103515245 + 12345) & MASK32; sx = 2 + (state % (width // 5 if width // 5 > 3 else 3))
        state = (state * 1103515245 + 12345) & MASK32; sy = 2 + (state % (height // 5 if height // 5 > 3 else 3))
        state = (state * 1103515245 + 12345) & MASK32
        ang = (state % 6283) / 1000.0 - 3.1415
        ca = cos_t(ang)
        sa = sin_t(ang)

        for y in range(height):
            for x in range(width):
                rx = ca * (x - cx) - sa * (y - cy)
                ry = sa * (x - cx) + ca * (y - cy)
                if (s & 1) == 0:
                    hit = (rx * rx) / (sx * sx) + (ry * ry) / (sy * sy) <= 1.0
                else:
                    hit = (-sx <= rx <= sx) and (-sy <= ry <= sy)
                if hit:
                    grid[y * width + x] += 1

    for i in range(size):
        if grid[i] > 0: covered += 1
        if grid[i] > 1: overlap += 1
        checksum = (checksum + <unsigned int>(grid[i] * (i + 1))) & MASK32

    free(grid)
    return (covered, overlap, checksum)
