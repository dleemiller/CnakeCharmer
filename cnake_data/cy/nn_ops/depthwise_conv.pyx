# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Depthwise 1D convolution on f32 tensor (basic Cython, scalar loop).

Keywords: depthwise_conv, convolution, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def depthwise_conv(int n):
    """Depthwise 1D convolution, return sum of output."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef int kernel_size = 3
    cdef int out_spatial = spatial - kernel_size + 1

    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *kernel = <float *>malloc(channels * kernel_size * sizeof(float))
    if not inp or not kernel:
        raise MemoryError()

    cdef int c, s, k, inp_offset
    cdef double total = 0.0
    cdef float val

    # Generate input
    for c in range(n):
        inp[c] = sin(c * 0.01) * 10.0

    # Generate kernel
    for c in range(channels):
        for k in range(kernel_size):
            kernel[c * kernel_size + k] = sin((c * kernel_size + k) * 0.5) * 0.5

    # Depthwise conv
    for c in range(channels):
        inp_offset = c * spatial
        for s in range(out_spatial):
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kernel[c * kernel_size + k]
            total += val

    free(inp)
    free(kernel)
    return total
