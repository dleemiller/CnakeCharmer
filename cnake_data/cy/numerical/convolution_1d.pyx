# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D convolution of a deterministic signal with a fixed kernel (Cython-optimized).

Keywords: numerical, convolution, signal processing, 1D, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def convolution_1d(int n):
    """Compute 1D convolution using C arrays for signal, kernel, and output."""
    cdef int k_len = 5
    cdef int out_len = n - k_len + 1

    cdef double *signal = <double *>malloc(n * sizeof(double))
    cdef double *output = <double *>malloc(out_len * sizeof(double))
    cdef double kernel[5]
    if not signal or not output:
        if signal: free(signal)
        if output: free(output)
        raise MemoryError()

    cdef int i, j
    cdef double s

    kernel[0] = 0.1
    kernel[1] = 0.2
    kernel[2] = 0.4
    kernel[3] = 0.2
    kernel[4] = 0.1

    for i in range(n):
        signal[i] = (i * 7 + 3) % 100 / 10.0

    for i in range(out_len):
        s = 0.0
        for j in range(k_len):
            s += signal[i + j] * kernel[j]
        output[i] = s

    result = [output[i] for i in range(out_len)]
    free(signal)
    free(output)
    return result
