# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Elementwise multiplication of two f32 tensors.

Keywords: elementwise, multiply, neural network, tensor, f32, cython
"""

from cnake_data.benchmarks import cython_benchmark


cdef double _elementwise_mul_kernel(int n) noexcept nogil:
    cdef int i = 0
    cdef float a0, b0, a1, b1, a2, b2, a3, b3
    cdef double total = 0.0

    while i + 3 < n:
        a0 = ((i * 31 + 17) % 1000) * 0.01
        b0 = ((i * 13 + 7) % 500) * 0.01
        a1 = (((i + 1) * 31 + 17) % 1000) * 0.01
        b1 = (((i + 1) * 13 + 7) % 500) * 0.01
        a2 = (((i + 2) * 31 + 17) % 1000) * 0.01
        b2 = (((i + 2) * 13 + 7) % 500) * 0.01
        a3 = (((i + 3) * 31 + 17) % 1000) * 0.01
        b3 = (((i + 3) * 13 + 7) % 500) * 0.01
        total += (a0 * b0) + (a1 * b1) + (a2 * b2) + (a3 * b3)
        i += 4

    while i < n:
        a0 = ((i * 31 + 17) % 1000) * 0.01
        b0 = ((i * 13 + 7) % 500) * 0.01
        total += a0 * b0
        i += 1

    return total


@cython_benchmark(syntax="cy", args=(5000000,))
def elementwise_mul(int n):
    """Multiply two tensors element-wise and return sum."""
    cdef double total = 0.0

    with nogil:
        total = _elementwise_mul_kernel(n)

    return total
