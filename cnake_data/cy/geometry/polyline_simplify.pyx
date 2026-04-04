# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Douglas-Peucker polyline simplification.

Keywords: douglas peucker, polyline, simplification, geometry, line simplify, cython
"""

from libc.math cimport sin
from libc.stdlib cimport malloc, free
from libc.string cimport memset

from cnake_data.benchmarks import cython_benchmark


cdef inline double point_to_seg_dist_sq(double px, double py,
                                         double ax, double ay,
                                         double bx, double by) nogil:
    cdef double dx = bx - ax
    cdef double dy = by - ay
    cdef double t

    if dx == 0.0 and dy == 0.0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)

    if t < 0.0:
        dx = px - ax
        dy = py - ay
    elif t > 1.0:
        dx = px - bx
        dy = py - by
    else:
        dx = px - (ax + t * dx)
        dy = py - (ay + t * dy)

    return dx * dx + dy * dy


@cython_benchmark(syntax="cy", args=(5000,))
def polyline_simplify(int n):
    """Generate a noisy polyline of n points and simplify it.

    Args:
        n: Number of points in the polyline.

    Returns:
        Tuple of (num_simplified, total_x, total_y, first_y).
    """
    cdef double tolerance_sq = 0.01 * 0.01
    cdef int i, first, last, index
    cdef double max_dist_sq, dist_sq, noise
    cdef int stack_top

    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    cdef char *markers = <char *>malloc(n * sizeof(char))
    # Stack for iterative DP (pairs of first, last)
    cdef int *stack_first = <int *>malloc(n * sizeof(int))
    cdef int *stack_last = <int *>malloc(n * sizeof(int))
    if not xs or not ys or not markers or not stack_first or not stack_last:
        free(xs)
        free(ys)
        free(markers)
        free(stack_first)
        free(stack_last)
        raise MemoryError()

    memset(markers, 0, n * sizeof(char))

    # Generate polyline
    cdef double inv_n = 1.0 / (n - 1) if n > 1 else 0.0
    for i in range(n):
        xs[i] = i * inv_n
        noise = ((i * 7 + 13) % 97) / 97.0 * 0.1 - 0.05
        ys[i] = sin(xs[i] * 6.283185307) * 0.5 + noise

    # Douglas-Peucker
    if n <= 2:
        for i in range(n):
            markers[i] = 1
    else:
        markers[0] = 1
        markers[n - 1] = 1
        stack_top = 0
        stack_first[0] = 0
        stack_last[0] = n - 1
        stack_top = 1

        while stack_top > 0:
            stack_top -= 1
            first = stack_first[stack_top]
            last = stack_last[stack_top]
            max_dist_sq = 0.0
            index = first

            for i in range(first + 1, last):
                dist_sq = point_to_seg_dist_sq(
                    xs[i], ys[i], xs[first], ys[first], xs[last], ys[last])
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    index = i

            if max_dist_sq > tolerance_sq:
                markers[index] = 1
                stack_first[stack_top] = first
                stack_last[stack_top] = index
                stack_top += 1
                stack_first[stack_top] = index
                stack_last[stack_top] = last
                stack_top += 1

    cdef int num_simplified = 0
    cdef double total_x = 0.0
    cdef double total_y = 0.0
    cdef double first_y = 0.0
    cdef int found_first = 0

    for i in range(n):
        if markers[i]:
            num_simplified += 1
            total_x += xs[i]
            total_y += ys[i]
            if not found_first:
                first_y = ys[i]
                found_first = 1

    free(xs)
    free(ys)
    free(markers)
    free(stack_first)
    free(stack_last)

    return (num_simplified, total_x, total_y, first_y)
