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

// ---- ReLU (vclamp with min=0, max=inf) ----
// XNNPACK f32-vclamp-avx pattern: load 8, max(zero), store 8
void xnn_relu_f32(size_t n, const float* input, float* output) {
    const __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;

    // Main loop: 16 floats per iteration
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);
        v0 = _mm256_max_ps(vzero, v0);
        v1 = _mm256_max_ps(vzero, v1);
        _mm256_storeu_ps(output + i, v0);
        _mm256_storeu_ps(output + i + 8, v1);
    }
    // 8-wide remainder
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_max_ps(vzero, v);
        _mm256_storeu_ps(output + i, v);
    }
    // Scalar remainder
    for (; i < n; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// ---- GEMM 4x8 microkernel (FMA3, Haswell 2013+) ----
// XNNPACK f32-gemm-4x8-minmax-fma3-broadcast pattern
void xnn_gemm_f32(size_t m, size_t n, size_t k,
                  const float* a, const float* b, float* c) {
    memset(c, 0, m * n * sizeof(float));

    size_t i, j, kk;
    for (i = 0; i + 4 <= m; i += 4) {
        for (j = 0; j + 8 <= n; j += 8) {
            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();
            __m256 vacc2 = _mm256_setzero_ps();
            __m256 vacc3 = _mm256_setzero_ps();

            for (kk = 0; kk < k; kk++) {
                __m256 va0 = _mm256_broadcast_ss(&a[(i+0)*k + kk]);
                __m256 va1 = _mm256_broadcast_ss(&a[(i+1)*k + kk]);
                __m256 va2 = _mm256_broadcast_ss(&a[(i+2)*k + kk]);
                __m256 va3 = _mm256_broadcast_ss(&a[(i+3)*k + kk]);
                __m256 vb  = _mm256_loadu_ps(&b[kk*n + j]);

                vacc0 = _mm256_fmadd_ps(va0, vb, vacc0);
                vacc1 = _mm256_fmadd_ps(va1, vb, vacc1);
                vacc2 = _mm256_fmadd_ps(va2, vb, vacc2);
                vacc3 = _mm256_fmadd_ps(va3, vb, vacc3);
            }

            // Add to existing C values
            __m256 c0 = _mm256_loadu_ps(&c[(i+0)*n + j]);
            __m256 c1 = _mm256_loadu_ps(&c[(i+1)*n + j]);
            __m256 c2 = _mm256_loadu_ps(&c[(i+2)*n + j]);
            __m256 c3 = _mm256_loadu_ps(&c[(i+3)*n + j]);
            _mm256_storeu_ps(&c[(i+0)*n + j], _mm256_add_ps(c0, vacc0));
            _mm256_storeu_ps(&c[(i+1)*n + j], _mm256_add_ps(c1, vacc1));
            _mm256_storeu_ps(&c[(i+2)*n + j], _mm256_add_ps(c2, vacc2));
            _mm256_storeu_ps(&c[(i+3)*n + j], _mm256_add_ps(c3, vacc3));
        }
        // Scalar remainder columns
        for (j = (n/8)*8; j < n; j++) {
            for (kk = 0; kk < k; kk++) {
                c[(i+0)*n+j] += a[(i+0)*k+kk] * b[kk*n+j];
                c[(i+1)*n+j] += a[(i+1)*k+kk] * b[kk*n+j];
                c[(i+2)*n+j] += a[(i+2)*k+kk] * b[kk*n+j];
                c[(i+3)*n+j] += a[(i+3)*k+kk] * b[kk*n+j];
            }
        }
    }
    // Remainder rows
    for (; i < m; i++) {
        for (kk = 0; kk < k; kk++) {
            for (j = 0; j < n; j++) {
                c[i*n+j] += a[i*k+kk] * b[kk*n+j];
            }
        }
    }
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


def benchmark_kernel_only(lib, relu_n=5000000, gemm_n=200):
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
    benchmark_kernel_only(lib)
    write_kernel_report(lib)

    print(f"\n{'=' * 60}")
    print("Done.")
