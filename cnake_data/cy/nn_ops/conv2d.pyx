# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D convolution with edge detection kernel.

Keywords: convolution, 2d, edge detection, image, neural network, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def conv2d(int n):
    """Convolve n x n image with 3x3 edge detection kernel, return sum."""
    cdef int kernel[9]
    kernel[0] = -1; kernel[1] = -1; kernel[2] = -1
    kernel[3] = -1; kernel[4] = 8;  kernel[5] = -1
    kernel[6] = -1; kernel[7] = -1; kernel[8] = -1

    cdef long long total = 0
    cdef int i, j, ki, kj, pixel, s
    cdef int row, col

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    row = i - 1 + ki
                    col = j - 1 + kj
                    pixel = (row * 17 + col * 31 + 5) % 256
                    s += pixel * kernel[ki * 3 + kj]
            total += s
    return int(total)
