# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Python-callable wrappers around engine kernels for benchmarking.

Handles allocation outside the timing loop so pure nogil kernels
can be measured in isolation. Used by run_benchmarks.py for the
kernel-only table and by scripts/compare_xnnpack.py.
"""

import platform
import time

from libc.stdlib cimport malloc, free
from libc.math cimport sin

from cnake_charmer.engine.kernels.relu_f32 cimport (
    relu_f32, relu_f32_avx, reduce_sum_f32, reduce_sum_f32_avx,
)
from cnake_charmer.engine.kernels.gemm_f32 cimport gemm_f32, gemm_f32_avx

# Platform detection
_ARCH = platform.machine()
_HAS_AVX = _ARCH in ("x86_64", "AMD64")
_HAS_NEON = _ARCH in ("aarch64", "arm64")

def get_simd_label():
    """Return the SIMD ISA label for this platform."""
    if _HAS_AVX:
        return "avx2+fma"
    elif _HAS_NEON:
        return "neon"
    return "none"


def bench_relu_kernel(int n, bint use_avx=True):
    """Pre-allocate tensor, time ONLY the kernel, return (time_ms, sum)."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

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

    start = time.perf_counter()
    if use_avx:
        gemm_f32_avx(A, B, C, n, n, n)
    else:
        gemm_f32(A, B, C, n, n, n)
    elapsed = time.perf_counter() - start

    cdef double trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return elapsed * 1000, trace


def bench_all_kernels():
    """Run all engine kernels (scalar + platform SIMD), return results.

    Returns list of dicts with keys:
        kernel, size, scalar_ms, simd_ms, simd_label
    simd_ms is None if no SIMD available for this platform.
    """
    results = []
    simd_label = get_simd_label()
    has_simd = _HAS_AVX or _HAS_NEON

    # ReLU
    relu_n = 5000000
    # Warmup
    bench_relu_kernel(1000, use_avx=False)
    scalar_ms, _ = bench_relu_kernel(relu_n, use_avx=False)

    simd_ms = None
    if has_simd:
        bench_relu_kernel(1000, use_avx=_HAS_AVX)
        simd_ms, _ = bench_relu_kernel(relu_n, use_avx=_HAS_AVX)

    results.append({
        "kernel": "relu",
        "size": f"{relu_n:,}",
        "scalar_ms": scalar_ms,
        "simd_ms": simd_ms,
        "simd_label": simd_label,
    })

    # GEMM
    gemm_n = 200
    bench_gemm_kernel(10, use_avx=False)
    scalar_ms, _ = bench_gemm_kernel(gemm_n, use_avx=False)

    simd_ms = None
    if has_simd:
        bench_gemm_kernel(10, use_avx=_HAS_AVX)
        simd_ms, _ = bench_gemm_kernel(gemm_n, use_avx=_HAS_AVX)

    results.append({
        "kernel": "gemm",
        "size": f"{gemm_n}x{gemm_n}",
        "scalar_ms": scalar_ms,
        "simd_ms": simd_ms,
        "simd_label": simd_label,
    })

    return results
