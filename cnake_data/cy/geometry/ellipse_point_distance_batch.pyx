# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Distance from points to ellipse boundary (Cython)."""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef inline double robust_length(double x0, double x1) noexcept nogil:
    cdef double ax = x0 if x0 >= 0 else -x0
    cdef double ay = x1 if x1 >= 0 else -x1
    if ax > ay:
        return ax * sqrt(1.0 + (ay / ax) * (ay / ax))
    return ay * sqrt(1.0 + (ax / ay) * (ax / ay))

cdef inline double get_root(double r0, double z0, double z1) noexcept nogil:
    cdef double n0 = r0 * z0
    cdef double s0 = z1 - 1.0
    cdef double s1 = robust_length(n0, z1)
    cdef double s, ratio0, ratio1, g, denom0, denom1
    cdef int k
    for k in range(100):
        s = 0.5 * (s0 + s1)
        if s == s0 or s == s1:
            return s
        denom0 = s + r0
        denom1 = s + 1.0
        if -1e-15 < denom1 < 1e-15:
            if denom1 >= 0.0:
                denom1 = 1e-15
            else:
                denom1 = -1e-15
        ratio0 = n0 / denom0
        ratio1 = z1 / denom1
        g = ratio0 * ratio0 + ratio1 * ratio1 - 1.0
        if g > 0.0:
            s0 = s
        elif g < 0.0:
            s1 = s
        else:
            return s
    return 0.5 * (s0 + s1)

cdef inline double distance_point_ellipse(double a, double b, double x, double y) noexcept nogil:
    cdef double z0, z1, g, r0, s, x0, y0, dx, dy, denom
    if x < 0: x = -x
    if y < 0: y = -y
    z0 = x / a
    z1 = y / b
    g = z0 * z0 + z1 * z1 - 1.0
    if g == 0.0:
        return 0.0
    r0 = sqrt(a / b)
    s = get_root(r0, z0, z1)
    x0 = r0 * x / (s + r0)
    denom = s + 1.0
    if -1e-15 < denom < 1e-15:
        if denom >= 0.0:
            denom = 1e-15
        else:
            denom = -1e-15
    y0 = y / denom
    dx = x - x0
    dy = y - y0
    return sqrt(dx * dx + dy * dy)


@cython_benchmark(syntax="cy", args=(6.0, 3.0, 50000, 23))
def ellipse_point_distance_batch(double a, double b, int n_points, int seed):
    cdef unsigned int state = <unsigned int>((seed * 1664525 + 1013904223) & MASK32)
    cdef int i, inside = 0
    cdef double x, y, d, total = 0.0, max_d = 0.0

    for i in range(n_points):
        state = (state * 1664525 + 1013904223) & MASK32
        x = (((<int>(state % 20001)) - 10000) / 10000.0) * (1.5 * a)
        state = (state * 1664525 + 1013904223) & MASK32
        y = (((<int>(state % 20001)) - 10000) / 10000.0) * (1.5 * b)
        if (x * x) / (a * a) + (y * y) / (b * b) <= 1.0:
            inside += 1
        d = distance_point_ellipse(a, b, x, y)
        total += d
        if d > max_d:
            max_d = d
    return (total, max_d, inside)
