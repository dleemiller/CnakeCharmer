# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.math cimport sqrt, cos, fabs
from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef double PI = 3.141592653589793


@cython_benchmark(syntax="cy", args=(200,))
def level_set_delta(int n, double eps=1.5):
    """Compute the smoothed Dirac delta on a 2D signed distance field."""
    cdef int x, y
    cdef double cx = n / 2.0
    cdef double cy_center = n / 2.0
    cdef double radius = n / 4.0
    cdef double phi, phi_x, phi_y, grad, d
    cdef double total = 0.0
    cdef double max_val = 0.0
    cdef int count = 0
    cdef double *a = <double *>malloc(n * n * sizeof(double))

    if not a:
        raise MemoryError()

    for x in range(n):
        for y in range(n):
            a[x * n + y] = sqrt((x - cx) * (x - cx) + (y - cy_center) * (y - cy_center)) - radius

    for x in range(1, n - 1):
        for y in range(1, n - 1):
            phi = a[x * n + y]
            if fabs(phi) < eps:
                phi_x = (a[(x + 1) * n + y] - a[(x - 1) * n + y]) / 2.0
                phi_y = (a[x * n + y + 1] - a[x * n + y - 1]) / 2.0
                grad = sqrt(phi_x * phi_x + phi_y * phi_y)
                d = grad / (2.0 * eps) * (1.0 + cos(phi * PI / eps))
                total += d
                if d > max_val:
                    max_val = d
                count += 1

    free(a)
    return (total, max_val, count)
