# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fused residual add + ReLU on f32 tensor (basic Cython, scalar loop).

output = relu(input + residual).

Keywords: residual, add, relu, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def residual_add(int n):
    """Fused residual add + ReLU, return sum."""
    cdef float *inp = <float *>malloc(n * sizeof(float))
    cdef float *residual = <float *>malloc(n * sizeof(float))
    if not inp or not residual:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float val

    # Generate input and residual
    for i in range(n):
        inp[i] = sin(i * 0.01) * 10.0
        residual[i] = cos(i * 0.01) * 10.0

    # Fused residual add + ReLU
    for i in range(n):
        val = inp[i] + residual[i]
        if val < 0.0:
            val = 0.0
        total += val

    free(inp)
    free(residual)
    return total
