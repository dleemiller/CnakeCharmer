# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find the minimum enclosing circle of n deterministic points (Cython).

Keywords: geometry, minimum enclosing circle, welzl, bounding circle, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def minimum_enclosing_circle(int n):
    """Find minimum enclosing circle using incremental algorithm."""
    cdef double *px = <double *>malloc(n * sizeof(double))
    cdef double *py = <double *>malloc(n * sizeof(double))
    if not px or not py:
        free(px); free(py)
        raise MemoryError()

    cdef int i, j, k
    cdef double cx, cy, r, dx, dy, dx2, dy2, dx3, dy3
    cdef double ax, ay, bx, by, ccx, ccy
    cdef double d, ux, uy, d1, d2, d3
    cdef double tmp
    cdef long long seed = 42

    # Generate points
    for i in range(n):
        px[i] = ((i * 73 + 11) % 997) - 498.5
        py[i] = ((i * 37 + 23) % 991) - 495.5

    # Deterministic Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        j = <int>(seed % (i + 1))
        tmp = px[i]; px[i] = px[j]; px[j] = tmp
        tmp = py[i]; py[i] = py[j]; py[j] = tmp

    cx = px[0]
    cy = py[0]
    r = 0.0

    for i in range(1, n):
        dx = px[i] - cx
        dy = py[i] - cy
        if dx * dx + dy * dy > (r + 1e-10) * (r + 1e-10):
            cx = px[i]
            cy = py[i]
            r = 0.0
            for j in range(i):
                dx2 = px[j] - cx
                dy2 = py[j] - cy
                if dx2 * dx2 + dy2 * dy2 > (r + 1e-10) * (r + 1e-10):
                    # Circle from two points i, j
                    cx = (px[i] + px[j]) * 0.5
                    cy = (py[i] + py[j]) * 0.5
                    dx2 = px[i] - px[j]
                    dy2 = py[i] - py[j]
                    r = sqrt(dx2 * dx2 + dy2 * dy2) * 0.5
                    for k in range(j):
                        dx3 = px[k] - cx
                        dy3 = py[k] - cy
                        if dx3 * dx3 + dy3 * dy3 > (r + 1e-10) * (r + 1e-10):
                            # Circle from three points i, j, k
                            ax = px[i]; ay = py[i]
                            bx = px[j]; by = py[j]
                            ccx = px[k]; ccy = py[k]
                            d = 2.0 * (ax * (by - ccy) + bx * (ccy - ay) + ccx * (ay - by))
                            if d > -1e-14 and d < 1e-14:
                                # Degenerate: pick largest pair
                                d1 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                                d2 = (bx - ccx) * (bx - ccx) + (by - ccy) * (by - ccy)
                                d3 = (ax - ccx) * (ax - ccx) + (ay - ccy) * (ay - ccy)
                                if d1 >= d2 and d1 >= d3:
                                    cx = (ax + bx) * 0.5
                                    cy = (ay + by) * 0.5
                                    r = sqrt(d1) * 0.5
                                elif d2 >= d3:
                                    cx = (bx + ccx) * 0.5
                                    cy = (by + ccy) * 0.5
                                    r = sqrt(d2) * 0.5
                                else:
                                    cx = (ax + ccx) * 0.5
                                    cy = (ay + ccy) * 0.5
                                    r = sqrt(d3) * 0.5
                            else:
                                ux = ((ax * ax + ay * ay) * (by - ccy) + (bx * bx + by * by) * (ccy - ay) + (ccx * ccx + ccy * ccy) * (ay - by)) / d
                                uy = ((ax * ax + ay * ay) * (ccx - bx) + (bx * bx + by * by) * (ax - ccx) + (ccx * ccx + ccy * ccy) * (bx - ax)) / d
                                dx3 = ax - ux
                                dy3 = ay - uy
                                cx = ux
                                cy = uy
                                r = sqrt(dx3 * dx3 + dy3 * dy3)

    free(px)
    free(py)
    return (cx, cy, r)
