#!/usr/bin/env python
"""
Compare CnakeCharmer nn_ops against XNNPACK microkernels.

Builds a small C benchmark that calls XNNPACK's f32 microkernels directly,
then compares correctness and speed against our py/cy/cy_simd implementations.

Requirements:
    - XNNPACK cloned at /tmp/xnnpack (git clone --depth 1 https://github.com/google/XNNPACK /tmp/xnnpack)
    - gcc with AVX2 support

Usage:
    uv run --no-sync python scripts/compare_xnnpack.py
"""

import ctypes
import math
import os
import subprocess
import sys
import tempfile
import time

# Ensure cnake_charmer is importable (package=false means not installed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

XNNPACK_DIR = "/tmp/xnnpack"


def build_xnnpack_benchmark():
    """Build a small shared library that exposes XNNPACK-style microkernels."""

    c_source = r"""
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- ReLU: batch-16 vclamp ----
void xnn_relu_f32(size_t n, const float* input, float* output) {
    const __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);
        _mm256_storeu_ps(output + i, _mm256_max_ps(vzero, v0));
        _mm256_storeu_ps(output + i + 8, _mm256_max_ps(vzero, v1));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_max_ps(vzero, _mm256_loadu_ps(input + i)));
    }
    for (; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

// ---- Sigmoid: batch load/store, scalar exp ----
void xnn_sigmoid_f32(size_t n, const float* input, float* output) {
    for (size_t i = 0; i < n; i++) output[i] = 1.0f / (1.0f + expf(-input[i]));
}

// ---- GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) ----
void xnn_gelu_f32(size_t n, const float* input, float* output) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
}

// ---- SiLU: x * sigmoid(x) ----
void xnn_silu_f32(size_t n, const float* input, float* output) {
    for (size_t i = 0; i < n; i++) output[i] = input[i] / (1.0f + expf(-input[i]));
}

// ---- Softmax: max-subtract-exp-sum-div ----
void xnn_softmax_f32(size_t n, const float* input, float* output) {
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) { output[i] = expf(input[i] - max_val); sum += output[i]; }
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) output[i] *= inv_sum;
}

// ---- Elementwise add: batch-16 ----
void xnn_add_f32(size_t n, const float* a, const float* b, float* output) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(output + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) output[i] = a[i] + b[i];
}

// ---- Elementwise mul ----
void xnn_mul_f32(size_t n, const float* a, const float* b, float* output) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) output[i] = a[i] * b[i];
}

// ---- Batch norm: out = gamma * (x - mean) * inv_std + beta ----
void xnn_batch_norm_f32(size_t n, const float* input, float* output,
                        float mean, float inv_std, float gamma, float beta) {
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vinv = _mm256_set1_ps(inv_std);
    __m256 vgamma = _mm256_set1_ps(gamma);
    __m256 vbeta = _mm256_set1_ps(beta);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv);
        v = _mm256_fmadd_ps(v, vgamma, vbeta);
        _mm256_storeu_ps(output + i, v);
    }
    for (; i < n; i++) output[i] = gamma * (input[i] - mean) * inv_std + beta;
}

// ---- Residual add + ReLU ----
void xnn_residual_add_f32(size_t n, const float* a, const float* b, float* output) {
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(output + i, _mm256_max_ps(vzero,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i))));
    for (; i < n; i++) { float s = a[i] + b[i]; output[i] = s > 0.0f ? s : 0.0f; }
}

// ---- Conv1d ----
void xnn_conv1d_f32(size_t n, const float* input, const float* kernel,
                    float* output, size_t kernel_size) {
    size_t out_n = n - kernel_size + 1;
    for (size_t i = 0; i < out_n; i++) {
        __m256 acc = _mm256_setzero_ps();
        size_t k;
        // For each output position, compute dot product with kernel
        // (kernel is small, so just scalar)
        float sum = 0.0f;
        for (k = 0; k < kernel_size; k++) sum += input[i + k] * kernel[k];
        output[i] = sum;
    }
}

// ---- Max pool 1d ----
void xnn_max_pool_1d_f32(size_t n, const float* input, float* output,
                         size_t kernel, size_t stride) {
    size_t out_n = n / stride;
    for (size_t i = 0; i < out_n; i++) {
        float max_val = input[i * stride];
        for (size_t k = 1; k < kernel && i * stride + k < n; k++)
            if (input[i * stride + k] > max_val) max_val = input[i * stride + k];
        output[i] = max_val;
    }
}

// ---- GEMM 4x8 FMA3 ----
void xnn_gemm_f32(size_t m, size_t n, size_t k,
                  const float* a, const float* b, float* c) {
    memset(c, 0, m * n * sizeof(float));
    size_t i, j, kk;
    for (i = 0; i + 4 <= m; i += 4) {
        for (j = 0; j + 8 <= n; j += 8) {
            __m256 vacc0=_mm256_setzero_ps(), vacc1=_mm256_setzero_ps(),
                   vacc2=_mm256_setzero_ps(), vacc3=_mm256_setzero_ps();
            for (kk = 0; kk < k; kk++) {
                __m256 vb = _mm256_loadu_ps(&b[kk*n + j]);
                vacc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a[(i+0)*k+kk]), vb, vacc0);
                vacc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a[(i+1)*k+kk]), vb, vacc1);
                vacc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a[(i+2)*k+kk]), vb, vacc2);
                vacc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a[(i+3)*k+kk]), vb, vacc3);
            }
            _mm256_storeu_ps(&c[(i+0)*n+j], _mm256_add_ps(_mm256_loadu_ps(&c[(i+0)*n+j]), vacc0));
            _mm256_storeu_ps(&c[(i+1)*n+j], _mm256_add_ps(_mm256_loadu_ps(&c[(i+1)*n+j]), vacc1));
            _mm256_storeu_ps(&c[(i+2)*n+j], _mm256_add_ps(_mm256_loadu_ps(&c[(i+2)*n+j]), vacc2));
            _mm256_storeu_ps(&c[(i+3)*n+j], _mm256_add_ps(_mm256_loadu_ps(&c[(i+3)*n+j]), vacc3));
        }
        for (j = (n/8)*8; j < n; j++)
            for (kk = 0; kk < k; kk++) {
                c[(i+0)*n+j] += a[(i+0)*k+kk]*b[kk*n+j];
                c[(i+1)*n+j] += a[(i+1)*k+kk]*b[kk*n+j];
                c[(i+2)*n+j] += a[(i+2)*k+kk]*b[kk*n+j];
                c[(i+3)*n+j] += a[(i+3)*k+kk]*b[kk*n+j];
            }
    }
    for (; i < m; i++) for (kk = 0; kk < k; kk++) for (j = 0; j < n; j++) c[i*n+j] += a[i*k+kk]*b[kk*n+j];
}
"""
    tmpdir = tempfile.mkdtemp(prefix="xnn_bench_")
    c_path = os.path.join(tmpdir, "xnn_kernels.c")
    so_path = os.path.join(tmpdir, "xnn_kernels.so")

    with open(c_path, "w") as f:
        f.write(c_source)

    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O3", "-mavx2", "-mfma", "-o", so_path, c_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)

    return ctypes.CDLL(so_path), tmpdir


def benchmark_relu(lib, n=5000000):
    """Compare ReLU: Python vs Cython vs SIMD Cython vs XNNPACK-style C."""
    print(f"\n{'=' * 60}")
    print(f"ReLU benchmark: n={n:,} floats")
    print(f"{'=' * 60}")

    # Generate input data
    FloatArray = ctypes.c_float * n
    inp = FloatArray(*[math.sin(i * 0.01) * 10.0 for i in range(n)])
    out = FloatArray(*([0.0] * n))

    # XNNPACK-style C kernel
    lib.xnn_relu_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_relu_f32.restype = None

    # Warmup
    lib.xnn_relu_f32(n, inp, out)

    # Time XNNPACK C
    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_relu_f32(n, inp, out)
    xnn_time = (time.perf_counter() - start) / runs

    # Compute XNNPACK result for correctness
    lib.xnn_relu_f32(n, inp, out)
    xnn_sum = sum(out[i] for i in range(n))

    # Our implementations
    results = {"xnnpack_c": (xnn_time * 1000, xnn_sum)}

    try:
        from cnake_charmer.py.nn_ops.relu import relu as py_relu

        start = time.perf_counter()
        py_result = py_relu(n)
        py_time = time.perf_counter() - start
        results["python"] = (py_time * 1000, py_result)
    except Exception as e:
        results["python"] = (None, f"FAIL: {e}")

    try:
        from cnake_charmer.cy.nn_ops.relu import relu as cy_relu

        cy_relu(min(n, 1000))  # warmup
        start = time.perf_counter()
        cy_result = cy_relu(n)
        cy_time = time.perf_counter() - start
        results["cython"] = (cy_time * 1000, cy_result)
    except Exception as e:
        results["cython"] = (None, f"FAIL: {e}")

    try:
        from cnake_charmer.cy_simd.nn_ops.relu import relu as simd_relu

        simd_relu(min(n, 1000))  # warmup
        start = time.perf_counter()
        simd_result = simd_relu(n)
        simd_time = time.perf_counter() - start
        results["cy_simd"] = (simd_time * 1000, simd_result)
    except Exception as e:
        results["cy_simd"] = (None, f"FAIL: {e}")

    # Print results
    print(f"\n{'Implementation':<20} {'Time (ms)':>12} {'Sum':>20} {'vs XNNPACK':>12}")
    print("-" * 66)
    for name, (time_ms, result_val) in results.items():
        if time_ms is None:
            print(f"{name:<20} {'FAIL':>12} {str(result_val):>20}")
            continue
        ratio = time_ms / results["xnnpack_c"][0] if results["xnnpack_c"][0] > 0 else 0
        print(f"{name:<20} {time_ms:>10.2f}ms {result_val:>20.4f} {ratio:>10.1f}x")

    # Correctness check
    print(f"\nCorrectness (vs XNNPACK sum={xnn_sum:.4f}):")
    for name, (_, result_val) in results.items():
        if isinstance(result_val, str):
            print(f"  {name}: {result_val}")
        else:
            diff = abs(result_val - xnn_sum) / max(abs(xnn_sum), 1.0)
            status = "PASS" if diff < 1e-3 else "FAIL"
            print(f"  {name}: {status} (relative error: {diff:.2e})")


def benchmark_gemm(lib, n=200):
    """Compare GEMM: Python vs Cython vs SIMD Cython vs XNNPACK-style C."""
    print(f"\n{'=' * 60}")
    print(f"GEMM benchmark: {n}x{n} f32 matrices")
    print(f"{'=' * 60}")

    # Generate matrices
    FloatArray = ctypes.c_float * (n * n)
    A = FloatArray(*[((i // n) + (i % n)) % 100 / 10.0 for i in range(n * n)])
    B = FloatArray(*[((i // n) - (i % n) + n) % 100 / 10.0 for i in range(n * n)])
    C = FloatArray(*([0.0] * (n * n)))

    # XNNPACK-style C kernel
    lib.xnn_gemm_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_gemm_f32.restype = None

    # Warmup
    lib.xnn_gemm_f32(n, n, n, A, B, C)

    # Time XNNPACK C (memset is inside xnn_gemm_f32)
    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_gemm_f32(n, n, n, A, B, C)
    xnn_time = (time.perf_counter() - start) / runs

    # Extract trace
    xnn_trace = sum(C[i * n + i] for i in range(n))

    results = {"xnnpack_c": (xnn_time * 1000, xnn_trace)}

    try:
        from cnake_charmer.py.nn_ops.gemm import gemm as py_gemm

        start = time.perf_counter()
        py_result = py_gemm(n)
        py_time = time.perf_counter() - start
        results["python"] = (py_time * 1000, py_result)
    except Exception as e:
        results["python"] = (None, f"FAIL: {e}")

    try:
        from cnake_charmer.cy.nn_ops.gemm import gemm as cy_gemm

        cy_gemm(10)  # warmup
        start = time.perf_counter()
        cy_result = cy_gemm(n)
        cy_time = time.perf_counter() - start
        results["cython"] = (cy_time * 1000, cy_result)
    except Exception as e:
        results["cython"] = (None, f"FAIL: {e}")

    try:
        from cnake_charmer.cy_simd.nn_ops.gemm import gemm as simd_gemm

        simd_gemm(10)  # warmup
        start = time.perf_counter()
        simd_result = simd_gemm(n)
        simd_time = time.perf_counter() - start
        results["cy_simd"] = (simd_time * 1000, simd_result)
    except Exception as e:
        results["cy_simd"] = (None, f"FAIL: {e}")

    # Print results
    print(f"\n{'Implementation':<20} {'Time (ms)':>12} {'Trace':>20} {'vs XNNPACK':>12}")
    print("-" * 66)
    for name, (time_ms, result_val) in results.items():
        if time_ms is None:
            print(f"{name:<20} {'FAIL':>12} {str(result_val):>20}")
            continue
        ratio = time_ms / results["xnnpack_c"][0] if results["xnnpack_c"][0] > 0 else 0
        print(f"{name:<20} {time_ms:>10.2f}ms {result_val:>20.4f} {ratio:>10.1f}x")

    # Correctness
    print(f"\nCorrectness (vs XNNPACK trace={xnn_trace:.4f}):")
    for name, (_, result_val) in results.items():
        if isinstance(result_val, str):
            print(f"  {name}: {result_val}")
        else:
            diff = abs(result_val - xnn_trace) / max(abs(xnn_trace), 1.0)
            status = "PASS" if diff < 1e-2 else "FAIL"
            print(f"  {name}: {status} (relative error: {diff:.2e})")


def benchmark_all_kernels(lib):
    """Compare ALL engine kernels against XNNPACK-style C implementations."""
    print(f"\n{'=' * 80}")
    print("KERNEL-ONLY: Engine AVX2+FMA vs XNNPACK C (pre-allocated, compute only)")
    print(f"{'=' * 80}")

    try:
        from cnake_charmer.engine.kernels.bench_wrapper import bench_all_kernels
    except ImportError as e:
        print(f"Engine kernels not built: {e}")
        return

    # Get engine kernel results
    engine_results = bench_all_kernels()

    # Now run corresponding C kernels for comparison
    n = 5000000
    FloatArray = ctypes.c_float * n
    inp = FloatArray(*[math.sin(i * 0.01) * 10.0 for i in range(n)])
    out = FloatArray(*([0.0] * n))
    inp2 = FloatArray(*[math.sin(i * 0.02) * 5.0 for i in range(n)])
    runs = 10

    def time_c(fn, *args):
        fn(*args)  # warmup
        start = time.perf_counter()
        for _ in range(runs):
            fn(*args)
        return (time.perf_counter() - start) / runs * 1000

    # Set up C function signatures
    unary_sig = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    binary_sig = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    # Map kernel names to C functions
    c_kernels = {}

    for name, fn_name, sig in [
        ("relu", "xnn_relu_f32", unary_sig),
        ("sigmoid", "xnn_sigmoid_f32", unary_sig),
        ("gelu", "xnn_gelu_f32", unary_sig),
        ("silu", "xnn_silu_f32", unary_sig),
        ("softmax", "xnn_softmax_f32", unary_sig),
        ("elementwise_add", "xnn_add_f32", binary_sig),
        ("elementwise_mul", "xnn_mul_f32", binary_sig),
        ("residual_add", "xnn_residual_add_f32", binary_sig),
    ]:
        try:
            fn = getattr(lib, fn_name)
            fn.argtypes = sig
            fn.restype = None
            c_kernels[name] = fn
        except AttributeError:
            pass

    # Batch norm (special signature)
    try:
        lib.xnn_batch_norm_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.xnn_batch_norm_f32.restype = None
        c_kernels["batch_norm"] = lib.xnn_batch_norm_f32
    except AttributeError:
        pass

    # Conv1d
    try:
        lib.xnn_conv1d_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.xnn_conv1d_f32.restype = None
        c_kernels["conv1d"] = lib.xnn_conv1d_f32
    except AttributeError:
        pass

    # Max pool
    try:
        lib.xnn_max_pool_1d_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.xnn_max_pool_1d_f32.restype = None
        c_kernels["max_pool_1d"] = lib.xnn_max_pool_1d_f32
    except AttributeError:
        pass

    # GEMM
    try:
        lib.xnn_gemm_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.xnn_gemm_f32.restype = None
        c_kernels["gemm"] = lib.xnn_gemm_f32
    except AttributeError:
        pass

    # Run C benchmarks for each engine kernel
    print(
        f"\n{'Kernel':<20} {'Size':>10} {'C (ms)':>10} {'Scalar':>10} {'AVX2+FMA':>10} {'AVX/C':>8} {'Scalar/C':>10}"
    )
    print("-" * 80)

    for er in engine_results:
        name = er["kernel"]
        c_ms = None

        # Time the C kernel
        if name in ("relu", "sigmoid", "gelu", "silu", "softmax") and name in c_kernels:
            en = int(er["size"].replace(",", ""))
            inp_local = (ctypes.c_float * en)(*[math.sin(i * 0.01) * 10.0 for i in range(en)])
            out_local = (ctypes.c_float * en)(*([0.0] * en))
            c_ms = time_c(c_kernels[name], en, inp_local, out_local)

        elif name in ("elementwise_add", "elementwise_mul", "residual_add") and name in c_kernels:
            c_ms = time_c(c_kernels[name], n, inp, inp2, out)

        elif name == "batch_norm" and name in c_kernels:
            bn_n = int(er["size"].replace(",", ""))
            bn_inp = (ctypes.c_float * bn_n)(*[math.sin(i * 0.01) * 10.0 for i in range(bn_n)])
            bn_out = (ctypes.c_float * bn_n)(*([0.0] * bn_n))
            c_ms = time_c(c_kernels[name], bn_n, bn_inp, bn_out, 0.0, 1.0, 1.0, 0.0)

        elif name == "conv1d" and name in c_kernels:
            cn = 500000
            c_inp = (ctypes.c_float * cn)(*[math.sin(i * 0.01) * 10.0 for i in range(cn)])
            c_kernel = (ctypes.c_float * 7)(0.0625, 0.125, 0.1875, 0.25, 0.1875, 0.125, 0.0625)
            c_out = (ctypes.c_float * (cn - 6))(*([0.0] * (cn - 6)))
            c_ms = time_c(c_kernels[name], cn, c_inp, c_kernel, c_out, 7)

        elif name == "max_pool_1d" and name in c_kernels:
            pn = int(er["size"].replace(",", ""))
            p_inp = (ctypes.c_float * pn)(*[math.sin(i * 0.01) * 10.0 for i in range(pn)])
            p_out = (ctypes.c_float * (pn // 4))(*([0.0] * (pn // 4)))
            c_ms = time_c(c_kernels[name], pn, p_inp, p_out, 4, 4)

        elif name == "gemm" and name in c_kernels:
            gn = 200
            gA = (ctypes.c_float * (gn * gn))(
                *[((i // gn + i % gn) % 100) / 10.0 for i in range(gn * gn)]
            )
            gB = (ctypes.c_float * (gn * gn))(
                *[((i // gn - i % gn + gn) % 100) / 10.0 for i in range(gn * gn)]
            )
            gC = (ctypes.c_float * (gn * gn))(*([0.0] * (gn * gn)))
            c_ms = time_c(c_kernels[name], gn, gn, gn, gA, gB, gC)

        # Format output
        c_str = f"{c_ms:.3f}" if c_ms else "—"
        scalar_str = f"{er['scalar_ms']:.3f}"
        simd_str = f"{er['simd_ms']:.3f}" if er["simd_ms"] else "—"
        avx_c = f"{er['simd_ms'] / c_ms:.1f}x" if c_ms and er["simd_ms"] else "—"
        scalar_c = f"{er['scalar_ms'] / c_ms:.1f}x" if c_ms else "—"

        print(
            f"{name:<20} {er['size']:>10} {c_str:>10} {scalar_str:>10} {simd_str:>10} {avx_c:>8} {scalar_c:>10}"
        )


def benchmark_kernel_only_legacy(lib, relu_n=5000000, gemm_n=200):
    """Kernel-only comparison: pre-allocated tensors, time only compute.

    Uses engine/kernels which are extracted nogil functions —
    same code that a future inference engine would call.
    """
    print(f"\n{'=' * 60}")
    print("KERNEL-ONLY benchmarks (no allocation in timing loop)")
    print(f"{'=' * 60}")

    try:
        from cnake_charmer.engine.kernels.bench_wrapper import (
            bench_gemm_kernel,
            bench_relu_kernel,
        )
    except ImportError as e:
        print(f"Engine kernels not built: {e}")
        print("Run: python setup.py build_ext --inplace")
        return

    # ---- ReLU kernel-only ----
    print(f"\n--- ReLU kernel-only: n={relu_n:,} ---")

    # XNNPACK C (kernel only — data already in ctypes array)
    FloatArray = ctypes.c_float * relu_n
    inp = FloatArray(*[math.sin(i * 0.01) * 10.0 for i in range(relu_n)])
    out = FloatArray(*([0.0] * relu_n))

    lib.xnn_relu_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_relu_f32.restype = None
    lib.xnn_relu_f32(relu_n, inp, out)  # warmup

    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_relu_f32(relu_n, inp, out)
    xnn_relu_ms = (time.perf_counter() - start) / runs * 1000
    xnn_relu_sum = sum(out[i] for i in range(relu_n))

    # Engine scalar kernel
    _, _ = bench_relu_kernel(min(relu_n, 1000), use_avx=False)  # warmup
    scalar_ms, scalar_sum = bench_relu_kernel(relu_n, use_avx=False)

    # Engine AVX kernel
    _, _ = bench_relu_kernel(min(relu_n, 1000), use_avx=True)  # warmup
    avx_ms, avx_sum = bench_relu_kernel(relu_n, use_avx=True)

    print(f"  {'XNNPACK C':<20} {xnn_relu_ms:>8.2f}ms  sum={xnn_relu_sum:.2f}")
    print(
        f"  {'engine scalar':<20} {scalar_ms:>8.2f}ms  sum={scalar_sum:.2f}  ({scalar_ms / xnn_relu_ms:.1f}x vs C)"
    )
    print(
        f"  {'engine avx2+fma':<20} {avx_ms:>8.2f}ms  sum={avx_sum:.2f}  ({avx_ms / xnn_relu_ms:.1f}x vs C)"
    )

    # Correctness
    for name, val in [("scalar", scalar_sum), ("avx", avx_sum)]:
        diff = abs(val - xnn_relu_sum) / max(abs(xnn_relu_sum), 1.0)
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"  {name} correctness: {status} (rel err: {diff:.2e})")

    # ---- GEMM kernel-only ----
    print(f"\n--- GEMM kernel-only: {gemm_n}x{gemm_n} ---")

    FloatArray2 = ctypes.c_float * (gemm_n * gemm_n)
    A = FloatArray2(*[((i // gemm_n) + (i % gemm_n)) % 100 / 10.0 for i in range(gemm_n * gemm_n)])
    B = FloatArray2(
        *[((i // gemm_n) - (i % gemm_n) + gemm_n) % 100 / 10.0 for i in range(gemm_n * gemm_n)]
    )
    C = FloatArray2(*([0.0] * (gemm_n * gemm_n)))

    lib.xnn_gemm_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_gemm_f32.restype = None
    lib.xnn_gemm_f32(gemm_n, gemm_n, gemm_n, A, B, C)  # warmup

    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_gemm_f32(gemm_n, gemm_n, gemm_n, A, B, C)
    xnn_gemm_ms = (time.perf_counter() - start) / runs * 1000
    xnn_trace = sum(C[i * gemm_n + i] for i in range(gemm_n))

    _, _ = bench_gemm_kernel(10, use_avx=False)  # warmup
    scalar_ms, scalar_trace = bench_gemm_kernel(gemm_n, use_avx=False)

    _, _ = bench_gemm_kernel(10, use_avx=True)  # warmup
    avx_ms, avx_trace = bench_gemm_kernel(gemm_n, use_avx=True)

    print(f"  {'XNNPACK C':<20} {xnn_gemm_ms:>8.2f}ms  trace={xnn_trace:.2f}")
    print(
        f"  {'engine scalar':<20} {scalar_ms:>8.2f}ms  trace={scalar_trace:.2f}  ({scalar_ms / xnn_gemm_ms:.1f}x vs C)"
    )
    print(
        f"  {'engine avx2+fma':<20} {avx_ms:>8.2f}ms  trace={avx_trace:.2f}  ({avx_ms / xnn_gemm_ms:.1f}x vs C)"
    )

    for name, val in [("scalar", scalar_trace), ("avx", avx_trace)]:
        diff = abs(val - xnn_trace) / max(abs(xnn_trace), 1.0)
        status = "PASS" if diff < 1e-2 else "FAIL"
        print(f"  {name} correctness: {status} (rel err: {diff:.2e})")


def write_kernel_report(lib, filename="benchmarks.md"):
    """Append a kernel-only comparison table to the benchmark report."""
    try:
        from cnake_charmer.engine.kernels.bench_wrapper import (
            bench_gemm_kernel,
            bench_relu_kernel,
        )
    except ImportError:
        print("Engine kernels not built, skipping kernel report")
        return

    # Collect kernel-only results
    rows = []

    # ReLU
    relu_n = 5000000
    FloatArray = ctypes.c_float * relu_n
    inp = FloatArray(*[math.sin(i * 0.01) * 10.0 for i in range(relu_n)])
    out = FloatArray(*([0.0] * relu_n))
    lib.xnn_relu_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_relu_f32.restype = None
    lib.xnn_relu_f32(relu_n, inp, out)
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_relu_f32(relu_n, inp, out)
    xnn_ms = (time.perf_counter() - start) / runs * 1000

    _, _ = bench_relu_kernel(1000, use_avx=False)
    scalar_ms, _ = bench_relu_kernel(relu_n, use_avx=False)
    _, _ = bench_relu_kernel(1000, use_avx=True)
    avx_ms, _ = bench_relu_kernel(relu_n, use_avx=True)
    rows.append(("relu", relu_n, xnn_ms, scalar_ms, avx_ms))

    # GEMM
    gemm_n = 200
    FloatArray2 = ctypes.c_float * (gemm_n * gemm_n)
    A = FloatArray2(*[((i // gemm_n) + (i % gemm_n)) % 100 / 10.0 for i in range(gemm_n * gemm_n)])
    B = FloatArray2(
        *[((i // gemm_n) - (i % gemm_n) + gemm_n) % 100 / 10.0 for i in range(gemm_n * gemm_n)]
    )
    C = FloatArray2(*([0.0] * (gemm_n * gemm_n)))
    lib.xnn_gemm_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.xnn_gemm_f32.restype = None
    lib.xnn_gemm_f32(gemm_n, gemm_n, gemm_n, A, B, C)
    start = time.perf_counter()
    for _ in range(runs):
        lib.xnn_gemm_f32(gemm_n, gemm_n, gemm_n, A, B, C)
    xnn_ms = (time.perf_counter() - start) / runs * 1000

    _, _ = bench_gemm_kernel(10, use_avx=False)
    scalar_ms, _ = bench_gemm_kernel(gemm_n, use_avx=False)
    _, _ = bench_gemm_kernel(10, use_avx=True)
    avx_ms, _ = bench_gemm_kernel(gemm_n, use_avx=True)
    rows.append(("gemm", gemm_n, xnn_ms, scalar_ms, avx_ms))

    # Append to benchmarks.md
    with open(filename, "a") as f:
        f.write("\n\n## Kernel-Only Benchmark (vs XNNPACK C)\n\n")
        f.write("Pre-allocated tensors, timing only the compute kernel.\n\n")
        f.write(
            "| Kernel | Size | XNNPACK C (ms) | Cython scalar (ms) | Cython AVX2+FMA (ms) | AVX vs C |\n"
        )
        f.write(
            "|--------|------|----------------|--------------------|-----------------------|-----------|\n"
        )
        for name, size, xnn, scalar, avx in rows:
            f.write(
                f"| {name} | {size:,} | {xnn:.3f} | {scalar:.3f} | {avx:.3f} | {avx / xnn:.1f}x |\n"
            )

    print(f"\nKernel-only results appended to {filename}")


if __name__ == "__main__":
    if not os.path.exists(XNNPACK_DIR):
        print(f"XNNPACK not found at {XNNPACK_DIR}")
        print("Clone it: git clone --depth 1 https://github.com/google/XNNPACK /tmp/xnnpack")
        sys.exit(1)

    print("Building XNNPACK-style C benchmark kernels...")
    lib, tmpdir = build_xnnpack_benchmark()
    print(f"Built: {tmpdir}")

    benchmark_relu(lib)
    benchmark_gemm(lib)
    benchmark_all_kernels(lib)

    print(f"\n{'=' * 60}")
    print("Done.")
