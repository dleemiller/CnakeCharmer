# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply a 3x3 sharpening filter to a deterministic image (Cython-optimized).

Keywords: image processing, sharpen, unsharp mask, convolution, kernel, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def sharpen_filter(int n):
    """Apply a 3x3 sharpening kernel to an n x n image using flat C arrays."""
    cdef int *image = <int *>malloc(n * n * sizeof(int))
    if not image:
        raise MemoryError()

    cdef int i, j, ki, kj, acc
    cdef long long total = 0
    cdef int clipped = 0

    # 3x3 sharpening kernel flattened
    cdef int kernel[9]
    kernel[0] = 0;  kernel[1] = -1; kernel[2] = 0
    kernel[3] = -1; kernel[4] = 5;  kernel[5] = -1
    kernel[6] = 0;  kernel[7] = -1; kernel[8] = 0

    # Generate image
    for i in range(n):
        for j in range(n):
            image[i * n + j] = (i * 17 + j * 31) % 256

    # Apply sharpening (skip 1-pixel border)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            acc = 0
            for ki in range(3):
                for kj in range(3):
                    acc += kernel[ki * 3 + kj] * image[(i + ki - 1) * n + (j + kj - 1)]
            # Clamp to [0, 255]
            if acc < 0:
                acc = 0
                clipped += 1
            elif acc > 255:
                acc = 255
                clipped += 1
            total += acc

    free(image)
    return (<int>total, clipped)
