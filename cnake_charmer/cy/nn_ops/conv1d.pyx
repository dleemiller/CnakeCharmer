# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D convolution of a signal with a smoothing kernel.

Keywords: convolution, 1d, signal processing, neural network, cnn, cython
"""

from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def conv1d(int n):
    """Convolve signal with kernel [1,2,3,4,3,2,1]/16 and return sum."""
    cdef double kernel[7]
    kernel[0] = 1.0 / 16.0
    kernel[1] = 2.0 / 16.0
    kernel[2] = 3.0 / 16.0
    kernel[3] = 4.0 / 16.0
    kernel[4] = 3.0 / 16.0
    kernel[5] = 2.0 / 16.0
    kernel[6] = 1.0 / 16.0

    cdef int k_len = 7
    cdef int out_len = n - k_len + 1
    cdef double total = 0.0
    cdef double s
    cdef int i, j

    for i in range(out_len):
        s = 0.0
        for j in range(k_len):
            s += sin((i + j) * 0.01) * 100.0 * kernel[j]
        total += s
    return total
