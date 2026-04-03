# Contributing: Adding Problems to the Dataset

This repo is a **living dataset** of parallel Python/Cython implementations. Each problem exists as a matched pair (Python + Cython) with equivalence tests and benchmarks. The training pipeline reads directly from the repo structure.

## Directory Structure

```
cnake_charmer/
  py/{category}/{name}.py       ← Pure Python implementation (training prompt)
  cy/{category}/{name}.pyx      ← Cython implementation (portable, scalar)
  cy_simd/{category}/{name}.pyx ← SIMD-optimized Cython (AVX2+FMA / NEON)
  pp/{category}/{name}.py       ← Pure Python Cython syntax (optional)
  engine/
    tensor.pxd                  ← TensorView struct (future inference engine)
    kernels/                    ← Extracted nogil kernels (shared by cy_simd + engine)
tests/
  {category}/test_{name}.py     ← Equivalence tests
```

Categories: `algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

> New category directories need `__init__.py` in `py/`, `cy/`, and `tests/`.

## Adding a Standard Problem

### 1. Write the Python implementation

Create `cnake_charmer/py/{category}/{name}.py`:

```python
"""Brief description.

Keywords: relevant, search, terms
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))  # should take ~10-500ms in Python
def func_name(n: int) -> return_type:
    """What this function does.

    Args:
        n: Input size.

    Returns:
        The result.
    """
    # Pure Python, no external deps, deterministic
    ...
```

### 2. Write the Cython implementation

Create `cnake_charmer/cy/{category}/{name}.pyx`:

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Brief description.

Keywords: relevant, search, terms, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp  # as needed
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))  # same args as Python
def func_name(int n):
    """What this function does."""
    cdef int i, j
    cdef double *arr = <double *>malloc(n * sizeof(double))
    if not arr:
        raise MemoryError()

    # C-optimized implementation ...

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
```

**Optimization techniques (in order of impact):**
- Use `cdef` for ALL loop variables and accumulators
- Use C arrays (`malloc`/`free`) instead of Python lists for hot loops
- Use `from libc.math cimport sqrt, exp, log` for math functions
- Use `from libc.string cimport memset, memcpy` for memory ops
- Use `cdivision=True` to avoid Python's zero-division checks
- Convert to Python objects only at the very end when returning
- For strings, work with `char *` or `bytes` instead of Python `str`
- Add `with nogil:` around pure-C loops to release the GIL

**Do NOT use:**
- `PyList_SET_ITEM` / `PyList_New` / `Py_INCREF` — causes segfaults. Use list comprehensions instead.
- `import cython` with annotation syntax (`cython.int`) in `.pyx` files — use native `cdef` syntax. The dataset loader strips `import cython`.
- Python `list.append()` in hot loops — pre-allocate with `malloc` or `[None] * n`.

**When the Python baseline uses NumPy:**

Don't replace vectorized NumPy calls with scalar Cython loops. NumPy's SIMD-optimized functions (`np.argmax`, `np.bincount`, `np.linalg.norm`, `np.interp`, etc.) will beat scalar C loops. Instead, demonstrate real-world Cython+NumPy synergy:

- **Fuse multiple NumPy operations** to eliminate temporary arrays (e.g., norm+scale+sum in one pass)
- **Custom reductions** NumPy can't express as a single vectorized call
- **Reduce Python overhead** between NumPy calls in a loop
- **Typed memoryview access** for operations that would need awkward NumPy indexing

Example — bad (slower than Python):
```cython
# DON'T: scalar loop replacing np.argmax
for i in range(n):
    for j in range(cols):
        if mat[i, j] > best_val: ...  # scalar comparison can't beat SIMD
```

Example — good (fuses 3 NumPy operations into 2 passes, zero temporaries):
```cython
# DO: fuse norm + scale + sum to avoid temp arrays
with nogil:
    for i in range(n):
        norm_sq = 0.0
        for j in range(cols):
            norm_sq += mat[i, j] * mat[i, j]
        inv_norm = 1.0 / sqrt(norm_sq)
        for j in range(cols):
            mat[i, j] *= inv_norm
            total += mat[i, j]
```

### 3. Choose a discriminating return value

Return values are how tests verify that py and cy implementations are equivalent. A good return value **changes when the algorithm is wrong**. A bad return value can accidentally match even with bugs.

**Preferred patterns (in order):**

1. **Return the full data structure** when n is small enough (n < ~10,000):
   ```python
   return sorted_array      # GOOD: any sorting bug changes the output
   return dp_table_row       # GOOD: any DP bug changes cell values
   ```

2. **Return a tuple of 2-4 independent indicators** when the full structure is too large:
   ```python
   return (total, dp[n//2])          # GOOD: samples an intermediate DP cell
   return (count, max_dist, reachable)  # GOOD: multiple independent checks
   ```

3. **Pair any aggregate with a sample or checksum** — never return a bare count/sum alone:
   ```python
   return count                        # BAD: broken search could still find same count
   return (count, last_match_pos)      # GOOD: position tracks where matches were found

   return total                        # BAD: errors in individual values can cancel out
   return (total, max_val, val_at_n_half)  # GOOD: spot-checks prevent cancellation

   return dp[n]                        # BAD: only checks the final cell
   return (dp[n], dp[n//2])            # GOOD: probes an intermediate subproblem
   ```

**Anti-patterns to avoid:**
- `return len(result)` — lengths are trivially predictable
- `return 1 if condition else 0` — boolean indicators are nearly useless
- `return n` or `return n - 1` — returning the input back
- `return sum(...)` alone — errors cancel out in sums

The same return structure must match exactly between py, cy, and cy_simd implementations.

### 4. Write the equivalence test

Create `tests/{category}/test_{name}.py`:

```python
"""Test {name} equivalence."""

import pytest

from cnake_charmer.py.{category}.{name} import func_name as py_func
from cnake_charmer.cy.{category}.{name} import func_name as cy_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_{name}_equivalence(n):
    # For floats: assert abs(py - cy) / max(abs(py), 1.0) < 1e-4
    # For ints/lists: assert py_result == cy_result
    assert py_func(n) == cy_func(n)
```

### 4. Compile, test, review annotations

```bash
uv run python setup.py build_ext --inplace
uv run pytest tests/{category}/test_{name}.py -v
```

**Review the HTML annotation** (`cnake_charmer/cy/{category}/{name}.html`):
- **Yellow lines** = Python interactions (slow) — minimize
- **White lines** = pure C (fast) — maximize
- Target score: **>0.85** (85% C lines)
- Common culprits: `list.append()`, Python list indexing, `sort()` on tuples, string ops

### 5. Run benchmarks and commit

```bash
uv run run_benchmarks.py    # parallel, hash-cached, only re-runs changed
cat benchmarks.md
git add cnake_charmer/py/ cnake_charmer/cy/ tests/ benchmarks.md .benchmark_cache.json
```

---

## Adding nn_ops (Neural Network Operations)

The `nn_ops` category is special — these are building blocks for a future CPU inference engine (like XNNPACK). They follow a different pattern:

### Three-tier architecture

| Tier | Directory | Purpose | Compiler flags |
|------|-----------|---------|----------------|
| Python | `py/nn_ops/` | Naive baseline | standard |
| Portable Cython | `cy/nn_ops/` | Scalar C arrays, no SIMD | `-mavx2 -mfma -O3`* |
| SIMD Cython | `cy_simd/nn_ops/` | AVX2+FMA intrinsics | `-mavx2 -mfma -O3` |

\* nn_ops in `cy/` also get SIMD flags since the compiler may auto-vectorize scalar loops.

### Design for inference

nn_ops should be written as if they'll be used in a real inference engine:

1. **Use `float` (f32)**, not `double` or `int` — matches ML inference precision
2. **Operate on pre-allocated tensors** — the benchmark wrapper handles allocation, the kernel takes pointers
3. **Extract the kernel as a `cdef void ... noexcept nogil` function** in `engine/kernels/` so it can be shared between the benchmark and a future engine

### Engine kernel pattern

The core compute goes in `engine/kernels/{name}_f32.pxd` / `.pyx`:

```cython
# engine/kernels/relu_f32.pxd
cdef void relu_f32(const float *inp, float *out, int n) noexcept nogil
cdef void relu_f32_avx(const float *inp, float *out, int n) noexcept nogil
```

The benchmark wrapper (`cy_simd/nn_ops/relu.pyx`) calls these via `cimport`. A future inference engine graph executor would call the same kernels.

### SIMD intrinsics pattern (XNNPACK-style)

```cython
cdef extern from "immintrin.h" nogil:
    ctypedef float __m256
    __m256 _mm256_loadu_ps(const float *mem) noexcept
    __m256 _mm256_max_ps(__m256 a, __m256 b) noexcept
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) noexcept
    # ... all intrinsics must be declared noexcept nogil
```

Key patterns from XNNPACK:
- **ReLU (vclamp)**: `_mm256_max_ps(zero, x)` — batch 16 floats per iteration
- **GEMM (4x8 microkernel)**: broadcast A, load packed B, `_mm256_fmadd_ps` — 4 row accumulators × 8-wide
- **Remainder handling**: scalar fallback for elements not divisible by 8

### Hardware floor

**FMA3 (Haswell 2013+)** is the minimum target. All SIMD kernels can use `_mm256_fmadd_ps`. NEON support (ARM) is planned for future — kernel files will split by architecture when added.

### f32 precision in tests

nn_ops use `float` (32-bit) internally but Python uses `double` (64-bit). Tests need relative tolerance:

```python
assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
```

### Benchmarks

The benchmark report has two sections:
1. **Full Operation** — includes tensor allocation + compute + reduce (standard benchmark table)
2. **Kernel-Only (Inference Mode)** — pre-allocated tensors, times only the compute kernel. Compares portable Cython vs SIMD. Generated automatically by `run_benchmarks.py`.

---

## Finding New Problems

### Stack v2 DuckDB — Data Source & Tracking

The primary source for new problems is **The Stack v2 (dedup, Cython subset)**, stored in DuckDB:

| File | Size | Description |
|------|------|-------------|
| `utils/stack_data/stack_cython_full.duckdb` | 844 MB | Full dataset, 52,525 Cython files with downloaded content |
| `utils/stack_data/stack_cython_1k.duckdb` | 8.6 MB | 1k sample (for quick exploration) |
| `utils/stack_data/the-stack-v2-dedup-cython.parquet` | 15 MB | Raw parquet (metadata only, no content) |

#### Partition tracking (`split` column)

The `split` column in `stack_cython_full.duckdb` tracks which rows have been assigned to training splits:

| `split` value | Count | Meaning |
|---------------|-------|---------|
| `sft` | 64 | Already converted to py/cy problem pairs and included in SFT dataset |
| `sft_candidate` | 651 | Selected for SFT conversion, **not yet processed** |
| `grpo_candidate` | 2,235 | Selected for GRPO training (plain Python problems) |
| `NULL` | 49,575 | Unpartitioned — available for future selection |

**When you process a candidate into a problem pair, update its split from `sft_candidate` to `sft`:**

```sql
UPDATE stack_cython SET split = 'sft' WHERE blob_id = '<blob_id>';
```

#### Querying candidates

```bash
uv run --no-sync python -c "
import duckdb
con = duckdb.connect('utils/stack_data/stack_cython_full.duckdb', read_only=True)
rows = con.execute('''
    SELECT blob_id, filename, path, content, length_bytes
    FROM stack_cython
    WHERE split = 'sft_candidate'
    ORDER BY length_bytes ASC LIMIT 30
''').fetchall()
for bid, fn, path, content, size in rows:
    defs = [l.strip() for l in content.split(chr(10)) if l.strip().startswith(('def ','cpdef '))]
    print(f'{size:5d}B  {fn:30s}  {defs[0][:80] if defs else \"(no def)\"}')
con.close()
"
```

**Look for:** `libc.math` functions, numerical loops, `malloc`/`free` patterns, memoryviews.
**Skip:** C/C++ library wrappers, `cdef class`, multi-module imports, prange/OpenMP (Python baseline is single-threaded).

SFT candidates range from 415B to 8,000B (median ~4KB).

### Adapting Stack code

1. Read the source, identify the core algorithm
2. Write a pure Python version (self-contained, deterministic)
3. Write a Cython version using our conventions
4. Parameterize by `n`, tune benchmark args to 10-500ms Python time
5. Test, annotate, benchmark, commit
6. **Update the DuckDB split**: mark the row as `sft` (see above)

## Memory Safety (ASan)

The scoring pipeline includes AddressSanitizer (ASan) checking to detect memory errors in Cython code. This catches:

- **Memory leaks** — `malloc` without corresponding `free`
- **Buffer overflows** — writing past allocated bounds
- **Use-after-free** — accessing memory after `free`
- **Double-free** — calling `free` twice on the same pointer

### How it works

1. The `.pyx` file is compiled with `-fsanitize=address -fno-omit-frame-pointer`
2. The function is run with small inputs (`PYTHONMALLOC=malloc` to avoid CPython false positives)
3. ASan output is parsed, filtering by function name to ignore CPython internal allocations
4. Score: 1.0 (clean) or 0.0 (any error detected)

### Reward weight

Memory safety is 15% of the composite reward:

| Signal | Weight |
|--------|--------|
| Correctness | 30% |
| Performance (speedup) | 25% |
| Annotations (C-level code) | 20% |
| Memory safety (ASan) | 15% |
| Lint (cython-lint) | 10% |

### Testing manually

Via MCP:
```
check_memory("cnake_charmer/cy/algorithms/max_flow.pyx", "max_flow", "(100,)")
```

Via Python:
```python
from cnake_charmer.validate.memory_safety import check_memory_safety
code = open("cnake_charmer/cy/algorithms/max_flow.pyx").read()
result = check_memory_safety(code, "max_flow", test_args=(100,))
print(result.score, result.errors)
```

---

## MCP Tools

The project includes an MCP server for AI-assisted development:

```bash
claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
```

Available tools:

| Tool | Description |
|------|-------------|
| `score_problem` | Full composite reward for a problem pair (compile + test + benchmark + annotate + ASan) |
| `list_problems` | List all problem pairs with status |
| `compile_file` | Compile a .pyx and check for errors |
| `annotate_file` | Compile with HTML annotations for optimization analysis |
| `check_memory` | Run AddressSanitizer to detect leaks, overflows, use-after-free |

Auto-detects SIMD flags for `cy_simd/` and `nn_ops/` files.
