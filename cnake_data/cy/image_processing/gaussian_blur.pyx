# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply 5x5 Gaussian blur to a deterministic image (Cython-optimized).

Keywords: image processing, gaussian blur, convolution, kernel, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(400,))
def gaussian_blur(int n):
    """Apply 5x5 Gaussian blur to an n x n image using flat C arrays."""
    cdef int *image = <int *>malloc(n * n * sizeof(int))
    if not image:
        raise MemoryError()

    cdef int i, j, ki, kj, acc
    cdef long long total = 0
    cdef int kernel_sum = 273

    # 5x5 Gaussian kernel flattened
    cdef int kernel[25]
    kernel[0] = 1;  kernel[1] = 4;  kernel[2] = 7;  kernel[3] = 4;  kernel[4] = 1
    kernel[5] = 4;  kernel[6] = 16; kernel[7] = 26; kernel[8] = 16; kernel[9] = 4
    kernel[10] = 7; kernel[11] = 26; kernel[12] = 41; kernel[13] = 26; kernel[14] = 7
    kernel[15] = 4; kernel[16] = 16; kernel[17] = 26; kernel[18] = 16; kernel[19] = 4
    kernel[20] = 1; kernel[21] = 4;  kernel[22] = 7;  kernel[23] = 4;  kernel[24] = 1

    # Generate image
    for i in range(n):
        for j in range(n):
            image[i * n + j] = (i * 7 + j * 13 + 3) % 256

    # Apply blur (skip 2-pixel border)
    for i in range(2, n - 2):
        for j in range(2, n - 2):
            acc = 0
            for ki in range(5):
                for kj in range(5):
                    acc += kernel[ki * 5 + kj] * image[(i + ki - 2) * n + (j + kj - 2)]
            total += acc / kernel_sum

    free(image)
    return <int>total
