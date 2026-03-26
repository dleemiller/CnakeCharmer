# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SiLU/Swish on f32 tensor (basic Cython, scalar loop).

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).

Keywords: silu, swish, activation, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def silu(int n):
    """Allocate f32 C array tensor, apply SiLU, return sum."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float x

    # Allocate tensor
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # SiLU: x / (1 + exp(-x))
    total = 0.0
    for i in range(n):
        x = data[i]
        total += x / (1.0 + exp(-<double>x))

    free(data)
    return total
