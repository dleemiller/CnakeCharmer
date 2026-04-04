# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply 3x3 box blur to an n x n grayscale image and return sum of blurred pixels (Cython-optimized).

Keywords: image processing, box blur, convolution, 2D, filter, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def box_blur(int n):
    """Apply 3x3 box blur to an n x n grayscale image using C arrays."""
    cdef int i, j, di, dj, s, size
    cdef long long total

    size = n * n
    cdef int *img = <int *>malloc(size * sizeof(int))
    if not img:
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            img[i * n + j] = (i * 7 + j * 13 + 3) % 256

    total = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            s = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    s += img[(i + di) * n + (j + dj)]
            total += s // 9

    free(img)
    return int(total)
