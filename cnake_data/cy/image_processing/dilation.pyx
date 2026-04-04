# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Binary morphological dilation with 3x3 kernel (Cython-optimized).

Keywords: image processing, morphological dilation, binary image, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def dilation(int n):
    """Apply binary morphological dilation with 3x3 kernel using flat C arrays."""
    cdef int i, j, di, dj, dilated, count
    cdef int size = n * n
    cdef int i_start, i_end, j_start, j_end

    cdef int *img = <int *>malloc(size * sizeof(int))
    if not img:
        raise MemoryError()

    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 10 == 0:
                img[i * n + j] = 1
            else:
                img[i * n + j] = 0

    count = 0
    for i in range(n):
        for j in range(n):
            dilated = 0
            if i > 0:
                i_start = i - 1
            else:
                i_start = 0
            if i < n - 1:
                i_end = i + 1
            else:
                i_end = n - 1
            if j > 0:
                j_start = j - 1
            else:
                j_start = 0
            if j < n - 1:
                j_end = j + 1
            else:
                j_end = n - 1
            for di in range(i_start, i_end + 1):
                for dj in range(j_start, j_end + 1):
                    if img[di * n + dj] == 1:
                        dilated = 1
                        break
                if dilated == 1:
                    break
            count += dilated

    free(img)
    return count
