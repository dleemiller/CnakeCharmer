# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Binary morphological erosion with 3x3 kernel (Cython-optimized).

Keywords: image processing, morphological erosion, binary image, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def erosion(int n):
    """Apply binary morphological erosion with 3x3 kernel on n x n image."""
    cdef int i, j, di, dj, eroded, count
    cdef int size = n * n

    cdef int *img = <int *>malloc(size * sizeof(int))
    if not img:
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            if (i * i + j * j) % 17 < 14:
                img[i * n + j] = 1
            else:
                img[i * n + j] = 0

    count = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            eroded = 1
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if img[(i + di) * n + (j + dj)] == 0:
                        eroded = 0
                        break
                if eroded == 0:
                    break
            count += eroded

    free(img)
    return count
