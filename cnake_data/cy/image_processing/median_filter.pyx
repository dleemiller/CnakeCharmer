# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply 3x3 median filter to grayscale image (Cython-optimized).

Keywords: image processing, median filter, grayscale, smoothing, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def median_filter(int n):
    """Apply 3x3 median filter to n x n grayscale image."""
    cdef int i, j, k, a, b, di, dj, key, total
    cdef int size = n * n
    cdef int window[9]

    cdef int *img = <int *>malloc(size * sizeof(int))
    cdef int *out = <int *>malloc(size * sizeof(int))
    if not img or not out:
        free(img); free(out)
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 17 + j * 31 + 5) % 256

    # Copy border pixels
    for i in range(n):
        out[i] = img[i]
        out[(n - 1) * n + i] = img[(n - 1) * n + i]
    for i in range(n):
        out[i * n] = img[i * n]
        out[i * n + n - 1] = img[i * n + n - 1]

    # Apply median filter to interior
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            k = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    window[k] = img[(i + di) * n + (j + dj)]
                    k += 1
            # Insertion sort on 9 elements
            for a in range(1, 9):
                key = window[a]
                b = a - 1
                while b >= 0 and window[b] > key:
                    window[b + 1] = window[b]
                    b -= 1
                window[b + 1] = key
            out[i * n + j] = window[4]

    total = 0
    for i in range(size):
        total += out[i]

    free(img)
    free(out)
    return total
