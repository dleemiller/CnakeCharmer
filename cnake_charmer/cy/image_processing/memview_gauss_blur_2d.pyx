# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Apply a 3x3 Gaussian blur to an n*n image using 2D typed memoryviews and return a checksum.

Keywords: image processing, gaussian blur, 2D, convolution, typed memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def memview_gauss_blur_2d(int n):
    """Apply 3x3 Gaussian blur using 2D memoryviews, return checksum."""
    cdef int i, j, di, dj
    cdef double val, checksum, ksum

    arr_img = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] img = arr_img

    arr_out = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] out = arr_out

    # Gaussian kernel weights
    cdef double kw[9]
    kw[0] = 1.0; kw[1] = 2.0; kw[2] = 1.0
    kw[3] = 2.0; kw[4] = 4.0; kw[5] = 2.0
    kw[6] = 1.0; kw[7] = 2.0; kw[8] = 1.0
    ksum = 16.0

    # Fill image
    for i in range(n):
        for j in range(n):
            img[i, j] = <double>((i * 73 + j * 59 + 11) % 256)

    # Initialize output to zero
    for i in range(n):
        for j in range(n):
            out[i, j] = 0.0

    # Apply blur to interior pixels
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            val = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    val += img[i + di, j + dj] * kw[(di + 1) * 3 + (dj + 1)]
            out[i, j] = val / ksum

    # Checksum
    checksum = 0.0
    for i in range(n):
        for j in range(n):
            checksum += out[i, j]

    return checksum
