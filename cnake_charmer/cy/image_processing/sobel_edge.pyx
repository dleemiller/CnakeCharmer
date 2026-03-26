# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply Sobel edge detection on an n x n grayscale image (Cython-optimized).

Keywords: image processing, Sobel, edge detection, gradient, convolution, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def sobel_edge(int n):
    """Apply Sobel edge detection and count pixels with gradient > 100."""
    cdef int i, j, size, count
    cdef int gx, gy
    cdef double mag

    size = n * n
    cdef int *img = <int *>malloc(size * sizeof(int))
    if not img:
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 7 + j * 13 + 3) % 256

    count = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            gx = (-img[(i - 1) * n + (j - 1)] + img[(i - 1) * n + (j + 1)]
                  - 2 * img[i * n + (j - 1)] + 2 * img[i * n + (j + 1)]
                  - img[(i + 1) * n + (j - 1)] + img[(i + 1) * n + (j + 1)])

            gy = (-img[(i - 1) * n + (j - 1)] - 2 * img[(i - 1) * n + j]
                  - img[(i - 1) * n + (j + 1)]
                  + img[(i + 1) * n + (j - 1)] + 2 * img[(i + 1) * n + j]
                  + img[(i + 1) * n + (j + 1)])

            mag = sqrt(<double>(gx * gx + gy * gy))
            if mag > 100.0:
                count += 1

    free(img)
    return count
