# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Python-callable wrappers around engine kernels for benchmarking.

These handle allocation and return values so the pure nogil kernels
can be timed from Python via compare_xnnpack.py.
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin

from cnake_charmer.engine.kernels.relu_f32 cimport (
    relu_f32, relu_f32_avx, reduce_sum_f32, reduce_sum_f32_avx,
)
from cnake_charmer.engine.kernels.gemm_f32 cimport gemm_f32, gemm_f32_avx


def bench_relu_kernel(int n, bint use_avx=True):
    """Pre-allocate tensor, time ONLY the kernel, return (time_ms, sum).

    Allocation happens before timing. Only the relu + reduce are timed.
    """
    import time

    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # Time kernel only
    cdef double total
    start = time.perf_counter()
    if use_avx:
        relu_f32_avx(data, data, n)
        total = reduce_sum_f32_avx(data, n)
    else:
        relu_f32(data, data, n)
        total = reduce_sum_f32(data, n)
    elapsed = time.perf_counter() - start

    free(data)
    return elapsed * 1000, total


def bench_gemm_kernel(int n, bint use_avx=True):
    """Pre-allocate matrices, time ONLY the kernel, return (time_ms, trace)."""
    import time

    cdef float *A = <float *>malloc(n * n * sizeof(float))
    cdef float *B = <float *>malloc(n * n * sizeof(float))
    cdef float *C = <float *>malloc(n * n * sizeof(float))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j
    for i in range(n):
        for j in range(n):
            A[i * n + j] = <float>((i + j) % 100) / 10.0
            B[i * n + j] = <float>((i - j + n) % 100) / 10.0

    # Time kernel only
    start = time.perf_counter()
    if use_avx:
        gemm_f32_avx(A, B, C, n, n, n)
    else:
        gemm_f32(A, B, C, n, n, n)
    elapsed = time.perf_counter() - start

    # Extract trace
    cdef double trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return elapsed * 1000, trace
