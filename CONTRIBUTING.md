# Contributing: Adding Problems to the Dataset

This repo is a **living dataset** of parallel Python/Cython implementations. Each problem exists as a matched pair (Python + Cython) with equivalence tests and benchmarks. The training pipeline reads directly from the repo structure.

## Directory Structure

```
cnake_charmer/
  py/{category}/{name}.py     ← Pure Python implementation (training prompt)
  cy/{category}/{name}.pyx    ← Cython implementation (ground truth baseline)
  pp/{category}/{name}.py     ← Pure Python Cython syntax (optional, third style)
tests/
  {category}/test_{name}.py   ← Equivalence tests
```

Categories: `algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

> **Note:** New category directories need an `__init__.py` in `py/`, `cy/`, and `tests/`.

## Adding a New Problem

### 1. Write the Python implementation

Create `cnake_charmer/py/{category}/{name}.py`:

```python
"""Brief description.

Keywords: relevant, search, terms
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))  # args for benchmarking (should take ~10-500ms)
def func_name(n: int) -> return_type:
    """What this function does.

    Args:
        n: Input size.

    Returns:
        The result.
    """
    # Pure Python implementation — no Cython, no C extensions
    # Must be deterministic (same input → same output)
    ...
```

**Guidelines:**
- Function must be self-contained (no external deps beyond stdlib)
- Must be deterministic — same `n` always produces the same output
- Choose benchmark args so Python version takes 10-500ms
- The function signature is what the model sees as its translation target

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

    # C-optimized implementation
    # ...

    # Convert to Python only at the end
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
- Use hardware-appropriate types: `int`, `long long`, `double`
- Convert to Python objects only at the very end when returning
- For strings, work with `char *` or `bytes` instead of Python `str`

**Do NOT use:**
- `PyList_SET_ITEM` / `PyList_New` / `Py_INCREF` — causes segfaults due to reference counting bugs. Use `[arr[i] for i in range(n)]` list comprehensions instead, which are safe and nearly as fast.
- `import cython` with annotation syntax (`cython.int`, `cython.double`) in `.pyx` files — use native `cdef` syntax instead. The dataset loader strips `import cython`.
- Python `list.append()` in hot loops — pre-allocate with `malloc` or `[None] * n`.

### 3. Write the equivalence test

Create `tests/{category}/test_{name}.py`:

```python
"""Test {name} equivalence."""

import pytest

from cnake_charmer.py.{category}.{name} import func_name as py_func
from cnake_charmer.cy.{category}.{name} import func_name as cy_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_{name}_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # For floats: assert abs(py_result - cy_result) < 1e-6
    # For float lists: assert all(abs(a-b) < 1e-6 for a,b in zip(py, cy))
    # For ints/lists: assert py_result == cy_result
    assert py_result == cy_result
```

### 4. Compile and test

```bash
# Build all Cython extensions
uv run python setup.py build_ext --inplace

# Run your test
uv run pytest tests/{category}/test_{name}.py -v

# Run all tests
uv run pytest tests/ -q
```

### 5. Review HTML annotations and optimize

**This is the most important step.** Every new problem should be optimized by reviewing the HTML annotations — not just the low performers.

Cython generates HTML annotation files showing which lines fall back to Python (yellow) vs run as pure C (white). The build step with `annotate=True` (default in `setup.py`) creates these at `cnake_charmer/cy/{category}/{name}.html`.

The project includes an MCP server with tools for scoring and annotating — see [MCP Tools](#mcp-tools-for-ai-assisted-development) below. You can also open the HTML directly:

```bash
xdg-open cnake_charmer/cy/{category}/{name}.html
```

**What to look for in the HTML:**
- **Yellow lines** = Python object interactions (SLOW) — minimize these
- **White lines** = pure C operations (FAST) — maximize these
- Click the `+` to expand and see the generated C code for each line
- Common yellow culprits:
  - `list.append()` → pre-allocate with `malloc` instead
  - `arr[i]` on a Python list → use C array pointer access
  - Calling Python functions in a loop → use `cdef` functions or `libc`
  - String operations → work with `char *` bytes
  - Python `sort()` on lists of tuples → use `libc.stdlib.qsort` with C structs

**Target annotation score: >0.85** (85% of lines should be pure C)

### 6. Check quality targets

Use the MCP tools or the CLI to verify the implementation meets these targets:

- `compiled`: must be True
- `correctness`: must be 1.0
- `annotations`: >0.85
- `speedup`: >5x (varies by problem type)
- `total reward`: >0.90

### 7. Run benchmarks and commit

```bash
# Run benchmarks (only re-runs changed problems via hash caching)
uv run python run_benchmarks.py

# Force re-run all benchmarks
uv run python run_benchmarks.py --all

# Check results
cat benchmarks.md

# Commit (include the benchmark cache so future runs skip unchanged)
git add cnake_charmer/py/{category}/{name}.py \
       cnake_charmer/cy/{category}/{name}.pyx \
       tests/{category}/test_{name}.py \
       benchmarks.md .benchmark_cache.json
git commit -m "Add {name} problem pair ({Nx speedup})"
```

## Finding New Problems to Implement

### From the Stack v2 DuckDB

The repo includes ~1,000 real-world Cython files from The Stack v2 in `utils/stack_data/stack_cython_1k.duckdb`. Browse it for inspiration:

```bash
# Find short, self-contained functions (best candidates for the dataset)
uv run python -c "
import duckdb
con = duckdb.connect('utils/stack_data/stack_cython_1k.duckdb', read_only=True)
rows = con.execute('''
    SELECT filename, path, content, length_bytes
    FROM stack_cython
    WHERE content IS NOT NULL
      AND length_bytes BETWEEN 200 AND 2000
      AND is_generated = false
      AND content LIKE '%def %'
    ORDER BY length_bytes ASC
    LIMIT 30
''').fetchall()
for fn, path, content, size in rows:
    defs = [l.strip() for l in content.split(chr(10)) if l.strip().startswith(('def ','cpdef '))]
    print(f'{size:5d}B  {fn:30s}  {defs[0][:80] if defs else \"(no def)\"}')
con.close()
"
```

**What to look for:**
- Functions using `libc.math` (trig, sqrt, exp) — great speedups from C math
- Numerical loops with `cdef double` / `cdef int` — classic Cython wins
- Algorithms using `malloc`/`free` for C arrays — shows real optimization patterns
- Functions with memoryviews (`double[:]`) — teaches typed array access

**What to skip:**
- Files wrapping C/C++ libraries (`cdef extern from`) — not self-contained
- Classes (`cdef class`) — focus on functions for now
- Files with many imports from other Cython modules — dependency issues

### From the algorithmic catalog

The `cnake_charmer/sources/algorithmic.py` loader reads from `data/problems.jsonl`. You can add problems there too:

```json
{"problem_id": "algo_042", "description": "Compute N-body gravitational forces", "python_code": "def nbody(n):\n    ...", "func_name": "nbody", "test_cases": [[[10]], [[50]]], "benchmark_args": [200], "category": "numerical", "difficulty": "hard"}
```

### Good problem categories to expand

| Category | What to look for | Example patterns from Stack |
|----------|-----------------|----------------------------|
| numerical | Pairwise distances, integration, FFT, interpolation | `libc.math` trig loops, memoryview dot products |
| algorithms | Graph algorithms, string matching, tree traversals | `malloc`-based adjacency lists, C array BFS |
| sorting | Heap sort, counting sort, Tim sort | In-place C array sorting with typed comparisons |
| dynamic_programming | Sequence alignment, path finding, optimization | 2D DP tables via flat `malloc` arrays |
| string_processing | Pattern matching, compression, encoding | `char *` iteration, byte-level comparison |
| math_problems | Number theory sieves, polynomial evaluation, GCD chains | Pure integer arithmetic in typed loops |

### Adapting Stack code into dataset pairs

When you find an interesting Cython file in the DuckDB:

1. **Read the full source**: `SELECT content FROM stack_cython WHERE filename = 'name.pyx'`
2. **Identify the core algorithm** — strip away imports, classes, and library wrappers
3. **Write a pure Python version** that's self-contained and deterministic
4. **Write a clean Cython version** using the patterns from the Stack code as inspiration (not copy-paste — adapt to our conventions)
5. **Make it parameterized by `n`** so benchmark args control the workload
6. Follow steps 3-7 above to test, annotate, score, and commit

## Optional: Pure Python Cython syntax (`pp/`)

The `pp/` directory uses Cython's [pure Python syntax](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html) — regular `.py` files that compile as Cython using decorators and annotations:

```python
import cython

@cython.cfunc
@cython.locals(i=cython.int, total=cython.double)
def helper(n: cython.int) -> cython.double:
    total = 0.0
    for i in range(n):
        total += i * 0.5
    return total
```

This is a newer syntax that runs as both Python and Cython. Add `pp/` implementations for problems where the pure Python syntax is a natural fit.

## MCP Tools for AI-Assisted Development

The project includes an MCP server with tools for compiling, annotating, and scoring Cython implementations. The tools are self-describing — AI assistants discover them automatically. Set it up with:

```bash
claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
```

These are the same validation and reward functions used during GRPO training.

## Hardware Instructions in Cython

Using hardware-level operations is explicitly in scope. Cython can access:

- **SIMD via C intrinsics**: `cdef extern from "immintrin.h"` for SSE/AVX
- **OpenMP parallelism**: `from cython.parallel cimport prange` with `nogil`
- **Cache-friendly patterns**: sequential memory access, struct-of-arrays
- **libc functions**: `memcpy`, `memset`, `qsort` — these compile to optimized machine code
- **Math intrinsics**: `libc.math` functions (`sqrt`, `exp`, `log`) use hardware FPU

The goal is to teach the model the full range of optimization techniques available in Cython, from basic type declarations to hardware-level optimization.
