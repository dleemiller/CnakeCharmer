# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D convolution filter for image processing.

Keywords: convolution, filter, image processing, kernel, 2d, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80,))
def convolution_2d(int n):
    """Apply a 3x3 convolution kernel to an n*n image.

    Args:
        n: Image dimension.

    Returns:
        Tuple of (total_sum, max_val, center_val).
    """
    cdef int kh = 3
    cdef int kw = 3
    cdef int pad = 1
    cdef int pn = n + 2 * pad
    cdef int i, j, ki, kj
    cdef double tot
    cdef double total_sum = 0.0
    cdef double max_val = 0.0
    cdef double center_val
    cdef int center = n // 2

    # Gaussian 3x3 kernel
    cdef double kernel[3][3]
    kernel[0][0] = 1.0 / 16; kernel[0][1] = 2.0 / 16; kernel[0][2] = 1.0 / 16
    kernel[1][0] = 2.0 / 16; kernel[1][1] = 4.0 / 16; kernel[1][2] = 2.0 / 16
    kernel[2][0] = 1.0 / 16; kernel[2][1] = 2.0 / 16; kernel[2][2] = 1.0 / 16

    cdef double *img = <double *>malloc(n * n * sizeof(double))
    cdef double *padded = <double *>malloc(pn * pn * sizeof(double))
    cdef double *result = <double *>malloc(n * n * sizeof(double))
    if not img or not padded or not result:
        free(img)
        free(padded)
        free(result)
        raise MemoryError()

    memset(padded, 0, pn * pn * sizeof(double))

    # Generate image
    for i in range(n):
        for j in range(n):
            img[i * n + j] = ((i * 7 + j * 13 + 3) % 97) / 97.0

    # Copy to padded
    for i in range(n):
        for j in range(n):
            padded[(i + pad) * pn + (j + pad)] = img[i * n + j]

    # Convolve
    with nogil:
        for i in range(n):
            for j in range(n):
                tot = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        tot += padded[(i + ki) * pn + (j + kj)] * kernel[ki][kj]
                result[i * n + j] = tot

        for i in range(n):
            for j in range(n):
                total_sum += result[i * n + j]
                if result[i * n + j] > max_val:
                    max_val = result[i * n + j]

    center_val = result[center * n + center]

    free(img)
    free(padded)
    free(result)
    return (total_sum, max_val, center_val)
