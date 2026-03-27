# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""GELU on f32 tensor (basic Cython, scalar loop).

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

Keywords: gelu, activation, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, tanh
from cnake_charmer.benchmarks import cython_benchmark

cdef float SQRT_2_OVER_PI = 0.7978845608028654
cdef float GELU_COEFF = 0.044715


@cython_benchmark(syntax="cy", args=(5000000,))
def gelu(int n):
    """Allocate f32 C array tensor, apply GELU, return sum."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float x, inner

    # Allocate tensor
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # GELU in-place
    for i in range(n):
        x = data[i]
        inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
        data[i] = 0.5 * x * (1.0 + tanh(inner))

    # Reduce
    for i in range(n):
        total += data[i]

    free(data)
    return total
