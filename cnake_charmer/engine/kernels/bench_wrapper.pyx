# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Python-callable wrappers around engine kernels for benchmarking.

Handles allocation outside the timing loop so pure nogil kernels
can be measured in isolation. Used by run_benchmarks.py for the
kernel-only table.
"""

import platform
import time

from libc.stdlib cimport malloc, free
from libc.math cimport sin

# Import engine kernels
from cnake_charmer.engine.kernels.relu_f32 cimport relu_f32, relu_f32_avx, reduce_sum_f32
from cnake_charmer.engine.kernels.gemm_f32 cimport gemm_f32, gemm_f32_avx
from cnake_charmer.engine.kernels.sigmoid_f32 cimport sigmoid_f32, sigmoid_f32_avx
from cnake_charmer.engine.kernels.gelu_f32 cimport gelu_f32, gelu_f32_avx
from cnake_charmer.engine.kernels.silu_f32 cimport silu_f32, silu_f32_avx
from cnake_charmer.engine.kernels.softmax_f32 cimport softmax_f32, softmax_f32_avx
from cnake_charmer.engine.kernels.batch_norm_f32 cimport batch_norm_f32, batch_norm_f32_avx
from cnake_charmer.engine.kernels.layer_norm_f32 cimport layer_norm_f32, layer_norm_f32_avx
from cnake_charmer.engine.kernels.elementwise_add_f32 cimport elementwise_add_f32, elementwise_add_f32_avx
from cnake_charmer.engine.kernels.elementwise_mul_f32 cimport elementwise_mul_f32, elementwise_mul_f32_avx
from cnake_charmer.engine.kernels.residual_add_f32 cimport residual_add_f32, residual_add_f32_avx
from cnake_charmer.engine.kernels.max_pool_1d_f32 cimport max_pool_1d_f32, max_pool_1d_f32_avx
from cnake_charmer.engine.kernels.conv1d_f32 cimport conv1d_f32, conv1d_f32_avx

# Platform detection
_ARCH = platform.machine()
_HAS_AVX = _ARCH in ("x86_64", "AMD64")
_HAS_NEON = _ARCH in ("aarch64", "arm64")


def get_simd_label():
    if _HAS_AVX: return "avx2+fma"
    elif _HAS_NEON: return "neon"
    return "none"


cdef void _fill(float *data, int n) noexcept:
    cdef int i
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0


cdef double _sum(const float *data, int n) noexcept:
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total


def bench_relu_kernel(int n, bint use_avx=True):
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data: raise MemoryError()
    _fill(data, n)
    start = time.perf_counter()
    if use_avx: relu_f32_avx(data, data, n)
    else: relu_f32(data, data, n)
    cdef double total = _sum(data, n)
    elapsed = time.perf_counter() - start
    free(data)
    return elapsed * 1000, total


def bench_gemm_kernel(int n, bint use_avx=True):
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
    if use_avx: gemm_f32_avx(A, B, C, n, n, n)
    else: gemm_f32(A, B, C, n, n, n)
    elapsed = time.perf_counter() - start
    cdef double trace = 0.0
    for i in range(n):
        trace += C[i * n + i]
    free(A); free(B); free(C)
    return elapsed * 1000, trace


def bench_all_kernels():
    """Run all engine kernels (scalar + platform SIMD), return results."""
    results = []
    simd_label = get_simd_label()
    has_simd = _HAS_AVX or _HAS_NEON

    cdef int n, i
    cdef float *inp
    cdef float *out
    cdef float *a
    cdef float *b
    cdef double scalar_ms, simd_ms_val

    # --- Unary ops: (inp, out, n) ---
    cdef list unary_specs = [("relu", 5000000), ("sigmoid", 2000000), ("gelu", 2000000),
                             ("silu", 2000000), ("softmax", 1000000)]
    for name, size in unary_specs:
        n = size
        inp = <float *>malloc(n * sizeof(float))
        out = <float *>malloc(n * sizeof(float))
        if not inp or not out:
            if inp: free(inp)
            if out: free(out)
            continue
        _fill(inp, n)

        start = time.perf_counter()
        if name == "relu": relu_f32(inp, out, n)
        elif name == "sigmoid": sigmoid_f32(inp, out, n)
        elif name == "gelu": gelu_f32(inp, out, n)
        elif name == "silu": silu_f32(inp, out, n)
        elif name == "softmax": softmax_f32(inp, out, n)
        scalar_ms = (time.perf_counter() - start) * 1000

        simd_ms = None
        if has_simd:
            _fill(inp, n)
            start = time.perf_counter()
            if name == "relu": relu_f32_avx(inp, out, n)
            elif name == "sigmoid": sigmoid_f32_avx(inp, out, n)
            elif name == "gelu": gelu_f32_avx(inp, out, n)
            elif name == "silu": silu_f32_avx(inp, out, n)
            elif name == "softmax": softmax_f32_avx(inp, out, n)
            simd_ms = (time.perf_counter() - start) * 1000

        free(inp); free(out)
        results.append({"kernel": name, "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})

    # --- Binary ops: (a, b, out, n) ---
    n = 5000000
    a = <float *>malloc(n * sizeof(float))
    b = <float *>malloc(n * sizeof(float))
    out = <float *>malloc(n * sizeof(float))
    if a and b and out:
        _fill(a, n)
        _fill(b, n)

        # elementwise_add
        start = time.perf_counter()
        elementwise_add_f32(a, b, out, n)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            elementwise_add_f32_avx(a, b, out, n)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "elementwise_add", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})

        # elementwise_mul
        start = time.perf_counter()
        elementwise_mul_f32(a, b, out, n)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            elementwise_mul_f32_avx(a, b, out, n)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "elementwise_mul", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})

        # residual_add (fused add + relu)
        start = time.perf_counter()
        residual_add_f32(a, b, out, n)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            residual_add_f32_avx(a, b, out, n)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "residual_add", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})

        free(a); free(b); free(out)

    # --- GEMM ---
    bench_gemm_kernel(10, use_avx=False)
    scalar_ms, _ = bench_gemm_kernel(200, use_avx=False)
    simd_ms = None
    if has_simd:
        bench_gemm_kernel(10, use_avx=True)
        simd_ms, _ = bench_gemm_kernel(200, use_avx=True)
    results.append({"kernel": "gemm", "size": "200x200", "scalar_ms": scalar_ms,
                    "simd_ms": simd_ms, "simd_label": simd_label})

    # --- Batch norm ---
    n = 5000000
    inp = <float *>malloc(n * sizeof(float))
    out = <float *>malloc(n * sizeof(float))
    if inp and out:
        _fill(inp, n)
        start = time.perf_counter()
        batch_norm_f32(inp, out, n, 0.0, 1.0, 1.0, 0.0)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            batch_norm_f32_avx(inp, out, n, 0.0, 1.0, 1.0, 0.0)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "batch_norm", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})
        free(inp); free(out)

    # --- Layer norm ---
    n = 1000000
    inp = <float *>malloc(n * sizeof(float))
    out = <float *>malloc(n * sizeof(float))
    if inp and out:
        _fill(inp, n)
        start = time.perf_counter()
        layer_norm_f32(inp, out, n, 1e-5)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            layer_norm_f32_avx(inp, out, n, 1e-5)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "layer_norm", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})
        free(inp); free(out)

    # --- Conv1d ---
    n = 500000
    cdef int ksize = 7
    cdef int out_n = n - ksize + 1
    inp = <float *>malloc(n * sizeof(float))
    out = <float *>malloc(out_n * sizeof(float))
    cdef float conv_k[7]
    conv_k[0] = 0.0625; conv_k[1] = 0.125; conv_k[2] = 0.1875
    conv_k[3] = 0.25; conv_k[4] = 0.1875; conv_k[5] = 0.125; conv_k[6] = 0.0625
    if inp and out:
        _fill(inp, n)
        start = time.perf_counter()
        conv1d_f32(inp, conv_k, out, n, ksize)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            conv1d_f32_avx(inp, conv_k, out, n, ksize)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "conv1d", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})
        free(inp); free(out)

    # --- Max pool 1d ---
    n = 5000000
    cdef int pk = 4
    cdef int ps = 4
    cdef int pout = n // ps
    inp = <float *>malloc(n * sizeof(float))
    out = <float *>malloc(pout * sizeof(float))
    if inp and out:
        _fill(inp, n)
        start = time.perf_counter()
        max_pool_1d_f32(inp, out, n, pk, ps)
        scalar_ms = (time.perf_counter() - start) * 1000
        simd_ms = None
        if has_simd:
            start = time.perf_counter()
            max_pool_1d_f32_avx(inp, out, n, pk, ps)
            simd_ms = (time.perf_counter() - start) * 1000
        results.append({"kernel": "max_pool_1d", "size": f"{n:,}", "scalar_ms": scalar_ms,
                        "simd_ms": simd_ms, "simd_label": simd_label})
        free(inp); free(out)

    return results
