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

**Vary function signatures** — this is training data, so problems should reflect the diversity of real-world code that gets converted to Cython. Don't make every function take a single `n: int`. Mix in realistic argument patterns:

| Pattern | Example | Cython benefit |
|---------|---------|----------------|
| Size parameter | `def sieve(n: int)` | Loop bounds as C int |
| Pre-built data | `def median(data: list)` | Typed memoryview or C array |
| Multiple params | `def lerp(a: float, b: float, steps: int)` | All args as cdef types |
| String input | `def hamming(s1: str, s2: str)` | `char *` / `bytes` processing |
| Nested structure | `def adjacency(edges: list, n: int)` | C arrays of C arrays |
| Class-based | `class Solver: def run(self, ...)` | `cdef class` with typed attrs |
| Config dict | `def simulate(params: dict)` | Unpack to cdef locals at entry |

**Hard rule for new additions:** avoid single-argument toy signatures like `def foo(n: int)` unless the real source problem is genuinely defined that way. Prefer multi-argument, data-driven interfaces (`seed/count/config`, arrays + shape parameters, thresholds, flags) that look like production code.

Before finalizing a new problem, quickly check:
- Does the function signature look like a realistic library/API call?
- Is `n` only one part of the configuration rather than the whole interface?
- Would this signature still make sense outside a benchmark harness?

## Non-Negotiable: No Templates / No Clones

The purpose of this dataset is to teach real code transformation, not pattern memorization.

**Templating is absolutely forbidden** for new problems:
- Do not mass-produce problems by copying one implementation and only renaming symbols, changing constants, or swapping minor arithmetic.
- Do not create families of near-identical problems across categories.
- Do not keep the same control flow/data model and just re-skin names (`*_class` variants with identical loops are not acceptable).

Every new problem must be **distinct and unique** in:
- Core algorithmic behavior (what it computes)
- Data layout / state transitions
- Failure modes and edge-case behavior
- Return-signal structure (what correctness evidence it exposes)

If two problems share more than superficial similarity, keep one and rewrite or delete the others.
When adapting Stack v2 candidates, preserve the original algorithmic intent of the source rather than forcing it into a reusable template shell.

**Class-based problems are encouraged.** Many real Cython conversions involve stateful objects — `cdef class` with typed attributes, `cdef` helper methods, and `__init__` that pre-allocates C arrays. The Stack v2 candidate pool includes class-based implementations that make good training examples. For classes:

```python
# py/ version — plain Python class
class ParticleSystem:
    def __init__(self, n):
        self.x = [0.0] * n
        self.v = [0.0] * n

    def step(self, dt):
        for i in range(len(self.x)):
            self.x[i] += self.v[i] * dt
```

```cython
# cy/ version — cdef class with C arrays
cdef class ParticleSystem:
    cdef double *x
    cdef double *v
    cdef int n

    def __cinit__(self, int n):
        self.n = n
        self.x = <double *>malloc(n * sizeof(double))
        self.v = <double *>malloc(n * sizeof(double))

    def __dealloc__(self):
        free(self.x)
        free(self.v)

    def step(self, double dt):
        cdef int i
        for i in range(self.n):
            self.x[i] += self.v[i] * dt
```

The benchmark decorator goes on a **factory function** that creates the object and calls its methods, so both py and cy versions use the same `@python_benchmark` / `@cython_benchmark` pattern.

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

**Decorator + `cpdef` note (practical guideline):**
- Cython may reject arbitrary decorators directly on `cpdef`/`cdef` functions in some setups.
- When that becomes an issue (or when it helps keep hot code clearly C-level), use a decorated `def` wrapper that calls a typed `cdef` core.
- This is a recommended pattern, not a hard rule. If direct `def`/`cpdef` structure is already clean and performant, keep it simple.

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

# For multi-arg functions, parametrize all args:
@pytest.mark.parametrize("s1,s2", [("abc", "axc"), ("kitten", "sitting")])
def test_{name}_strings(s1, s2):
    assert py_func(s1, s2) == cy_func(s1, s2)
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

### 5. Score with MCP and iterate

Run the full composite scoring to check quality:

```
score_problem("{category}/{name}")
```

**Minimum thresholds for acceptance:**

| Signal | Minimum | Target |
|--------|---------|--------|
| Total reward | **≥ 0.80** | > 0.90 |
| Correctness | **1.0** (required) | — |
| Speedup | **≥ 5x** | > 10x |
| Annotation | **≥ 0.85** | > 0.90 |
| Memory safety | **1.0** (required) | — |
| Lint | **1.0** (required) | — |

**Class/object coverage exception:** if a new problem intentionally preserves a class-heavy design (`class` / `cdef class`, stateful methods, object lifecycle) to improve dataset diversity, you may accept **high-0.8 annotation scores** (roughly `0.85-0.89`) when all of the following are true:
- `correctness = 1.0` (required)
- speedup is strong and meaningful (generally double-digit, often much higher)
- class/object structure is preserved on purpose (not flattened into purely procedural code just to raise annotation)

**If speedup is low (< 5x)**, check whether the Python baseline uses C-level builtins (`list.sort()`, `sum()`, `len()`) that already run in C. Replace with pure Python equivalents:
- `arr.sort()` → hand-written quicksort/mergesort in Python
- `sum(arr)` → explicit accumulation loop
- Built-in `sorted()` → manual sort

The Python version should be **idiomatic pure Python loops**, not wrappers around C-optimized builtins. This ensures the speedup reflects the real value of Cython optimization.

**If annotation score is low (< 0.85)**, check `annotate_file` for yellow lines:
- Add `cdef` to all loop variables and accumulators
- Replace Python `list` operations with `malloc`/`free` C arrays
- Add `with nogil:` around pure-C loop sections
- Use `from libc.math cimport ...` instead of Python `math` module
- Convert to Python objects only at the very end when returning

Iterate until the problem meets all thresholds before committing.

### 6. Run benchmarks and commit

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
2. Write a pure Python version (self-contained, deterministic, no C-level builtins in hot paths)
3. Write a Cython version using our conventions
4. Parameterize by `n`, tune benchmark args to 10-500ms Python time
5. Compile, test, then **run `score_problem` and iterate until ≥ 0.80 reward** (see Step 5 above)
6. Mark duplicates as `split = 'duplicate'` in DuckDB, skip them
7. **Update the DuckDB split**: mark converted rows as `sft`

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
from cnake_charmer.eval.memory_safety import check_memory_safety
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
For new Stack-to-triplet conversions, treat this as a required quality gate before marking done:
- Run `score_problem("{category}/{name}")`
- Require `correctness = 1.0`
- Require `annotation_score > 0.90`
- Require a meaningful speedup versus Python (typically >2x; if lower, document why the workload is already close to C/NumPy-limited)

For intentionally class-heavy conversions (to improve representation of object-oriented Cython patterns), use this adjusted gate:
- Require `correctness = 1.0`
- Require strong speedup versus Python
- Prefer `annotation_score > 0.90`, but allow high-`0.8x` when class/object structure is intentionally preserved.
