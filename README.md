# CnakeCharmer

A living dataset of parallel Python/Cython implementations for training AI models to translate Python → optimized Cython.

## Quickstart

```bash
# Install dependencies
uv sync

# Build Cython extensions
uv run python setup.py build_ext --inplace

# Run tests
uv run pytest tests/ -q

# Run benchmarks (parallel, hash-cached)
uv run run_benchmarks.py
```

> `uv sync` only installs Python dependencies — it does **not** compile Cython.
> Run `setup.py build_ext --inplace` when you add or change `.pyx` files.

## Project Goals

LLMs can write decent Python but struggle with efficient Cython. This is a training data gap.

This repo is both the **dataset** and the **training infrastructure**:

- **Dataset**: 523 matched Python/Cython pairs across 19 categories, version-controlled and CI-testable
- **Training**: Multi-turn GRPO with TRL GRPOTrainer — the model iteratively compiles, reviews HTML annotations, and optimizes its Cython output
- **nn_ops**: XNNPACK-style SIMD kernels (AVX2+FMA) for neural network operations, within 1.3x of hand-written C
- **Tools**: MCP server for AI-assisted development (compile, annotate, benchmark, score, memory safety via ASan)

## Project Structure

```
cnake_charmer/
  py/{category}/{name}.py       ← Pure Python (training prompt)
  cy/{category}/{name}.pyx      ← Portable Cython (scalar, ground truth)
  cy_simd/{category}/{name}.pyx ← SIMD-optimized Cython (AVX2+FMA)
  engine/
    tensor.pxd                  ← TensorView struct (future inference)
    kernels/                    ← Extracted nogil kernels (shared compute)
  validate/                     ← Compilation, annotation, correctness tools
  rewards/                      ← Reward functions for GRPO training
  training/                     ← TRL GRPOTrainer integration
  dataset/                      ← Loader that discovers pairs from repo
  mcp_server.py                 ← MCP server for Claude Code
tests/
  {category}/test_{name}.py     ← Equivalence tests
```

### Categories

`algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

### Three Tiers

For compute-intensive operations (e.g. `nn_ops`), three implementations exist:

| Tier | Directory | What it teaches |
|------|-----------|----------------|
| Python | `py/` | Naive baseline |
| Portable Cython | `cy/` | `cdef` types, C arrays, `libc.math`, extension types, memoryviews |
| SIMD Cython | `cy_simd/` | AVX2+FMA intrinsics, cache tiling, XNNPACK-style microkernels |

The SIMD kernels are extracted into `engine/kernels/` as `cdef void ... noexcept nogil` functions, ready for a future inference engine.

### Cython Feature Coverage

The problem set covers a broad range of Cython features beyond basic typed functions:

| Feature | Examples |
|---------|---------|
| `cdef class` (extension types) | `__cinit__`/`__dealloc__`, typed C attributes, `cdef`/`cpdef` methods |
| Special methods | `__getitem__`, `__setitem__`, `__len__`, `__contains__`, `__iter__`/`__next__`, `__add__`/`__mul__`/`__neg__`, `__richcmp__`, `__call__`, `__hash__` |
| Inheritance | `cdef class Child(Parent)` with `cpdef` method dispatch |
| Class decorators | `@cython.final`, `@cython.freelist`, `@cython.dataclasses.dataclass`, `cdef readonly`, `not None` |
| Properties | `@property` with getter and setter |
| `cdef enum` / `cpdef enum` | State machines, token types, direction enums, anonymous enums |
| Typed memoryviews | 1D/2D, C-contiguous `[::1]`, Fortran `[::1, :]`, `const`, `.T`, `.copy()`, `&view[0]`, `cython.view.array` |
| `cdef struct` / `cdef union` | Nested structs, packed structs, tagged unions, struct↔dict conversion, struct return |
| Fused types | `ctypedef fused` with type dispatch, memoryview params, type checking branches |
| `ctypedef` | Type aliases, function pointer typedefs |
| Function pointers | Dispatch tables, callbacks, `qsort` comparators |
| Buffer protocol | `__getbuffer__`/`__releasebuffer__` for 1D and 2D custom buffers |
| C-tuples | `(double, double)` return types from `cdef` functions |
| Error return specs | `except -1`, `except? -1.0`, `except *` |
| `cpdef` functions | Standalone module-level hybrid functions |
| `cdef extern from` | Direct C header access (`math.h`, `stdlib.h`, `string.h`) |
| C memory ops | `malloc`/`free`/`realloc`/`calloc`, `memcpy`/`memset`/`memcmp` |
| Stack arrays | Fixed-size `cdef int[1024]` on the stack |
| Forward declarations | `cdef class` forward declaration for recursive types |
| `prange` / `nogil` | OpenMP parallel loops, GIL release for C computation, schedule policies |
| NumPy interop | `cimport numpy`, typed memoryviews from arrays, `cnp.float64_t`, prange+NumPy |
| C++ interop | `libcpp.vector`/`map`/`set`/`unordered_map`, `cdef cppclass`, `except +`, `std::sort` templates, `enum class` |
| NumPy ufuncs | `@cython.ufunc` with scalar, fused-type, and integer-output ufuncs |
| NumPy + Pythran | `# cython: np_pythran=True` for fused NumPy expression templates |

See [FEATURE_COVERAGE.md](FEATURE_COVERAGE.md) for the full checklist (18/18 categories complete).

### Benchmarks

```bash
uv run run_benchmarks.py         # 4 parallel workers, hash caching
uv run run_benchmarks.py --all   # force re-run everything
uv run run_benchmarks.py -j 8    # 8 workers
```

Two benchmark tables in `benchmarks.md`:
1. **Full Operation** — allocation + compute + reduce (standard Python vs Cython comparison)
2. **Kernel-Only (Inference Mode)** — pre-allocated tensors, compute only. Compares portable Cython vs platform SIMD (AVX2+FMA on x86, NEON on ARM in future)

### XNNPACK Comparison

```bash
# Clone XNNPACK for reference comparison
git clone --depth 1 https://github.com/google/XNNPACK /tmp/xnnpack

# Run comparison (builds C microkernels, compares against our Cython)
uv run python scripts/compare_xnnpack.py
```

Current results: GEMM kernel within **1.3x** of hand-written C, ReLU kernel **matches C speed**.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide to adding new problem pairs and nn_ops.

## Training

The training pipeline uses TRL's GRPOTrainer with `environment_factory` for multi-turn tool-calling RL:

```python
from cnake_charmer.training.grpo import create_trainer
from cnake_charmer.dataset.loader import discover_pairs

trainer = create_trainer(
    model="Qwen/Qwen3-0.6B",
    problems=discover_pairs(),
)
trainer.train()
```

The model learns to call `compile`, `annotate`, `test`, and `benchmark` tools to iteratively improve its Cython output.

### Reward Signals

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Correctness | 30% | py/cy output equivalence across test cases |
| Performance | 25% | log-scaled speedup vs Python baseline |
| Annotations | 20% | Ratio of pure-C lines in Cython HTML annotations |
| Memory safety | 15% | AddressSanitizer (leaks, overflows, use-after-free) |
| Lint | 10% | cython-lint violations |
