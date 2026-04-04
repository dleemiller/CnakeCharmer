# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU on f32 tensor (basic Cython, scalar loop).

Same pattern as XNNPACK vclamp but without SIMD — scalar max(0,x).

Keywords: relu, activation, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def relu(int n):
    """Allocate f32 C array tensor, apply ReLU in-place, return sum."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0

    # Allocate tensor
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # ReLU in-place (scalar)
    for i in range(n):
        if data[i] < 0.0:
            data[i] = 0.0

    # Reduce
    for i in range(n):
        total += data[i]

    free(data)
    return total
