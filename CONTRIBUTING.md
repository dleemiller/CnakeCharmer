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

Categories: `algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`

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
- Consider SIMD-friendly patterns (sequential memory access, no branching)
- Convert to Python objects only at the very end when returning
- For strings, work with `char *` or `bytes` instead of Python `str`

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

# Run tests
uv run pytest tests/{category}/test_{name}.py -v

# Run all tests
uv run pytest tests/ -q
```

### 5. Review HTML annotations and optimize

This is the most important step. Cython generates HTML annotation files that show which lines fall back to Python (yellow) vs run as pure C (white).

```bash
# The build step with annotate=True (default in setup.py) generates HTML files
# Open the annotation: cnake_charmer/cy/{category}/{name}.html

# Or use our reward tools:
uv run python -c "
from cnake_charmer.validate.compiler import compile_cython
from cnake_charmer.validate.annotations import parse_annotations

code = open('cnake_charmer/cy/{category}/{name}.pyx').read()
result = compile_cython(code, annotate=True, keep_build=True)
ann = parse_annotations(html_path=result.html_path)
print(f'Score: {ann.score:.2f} ({ann.white_lines} C / {ann.total_lines} total)')
for h in ann.hints:
    print(f'  {h}')
"
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

**Target annotation score: >0.85** (85% of lines should be pure C)

### 6. Run the full reward analysis

```bash
uv run python -c "
from cnake_charmer.rewards.composite import composite_reward

# Load the Python reference
exec(open('cnake_charmer/py/{category}/{name}.py').read())

code = open('cnake_charmer/cy/{category}/{name}.pyx').read()
scores = composite_reward(
    cython_code=code,
    python_func=func_name,
    func_name='func_name',
    test_cases=[((10,),), ((100,),)],
    benchmark_args=(10000,),
)
for k, v in scores.items():
    print(f'{k}: {v}')
"
```

**Quality targets:**
- `compiled`: must be True
- `correctness`: must be 1.0
- `annotations`: >0.85
- `speedup`: >5x (varies by problem type)
- `total`: >0.90

### 7. Run benchmarks and commit

```bash
# Run benchmarks
uv run python run_benchmarks.py

# Check results
cat benchmarks.md

# Commit
git add cnake_charmer/py/{category}/{name}.py \
       cnake_charmer/cy/{category}/{name}.pyx \
       tests/{category}/test_{name}.py \
       benchmarks.md
git commit -m "Add {name} problem pair ({Nx speedup})"
```

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

The validation and reward functions are designed to work as MCP tools, so AI coding assistants (Claude Code, etc.) can call them during development:

| Tool | What it does |
|------|-------------|
| `compile(code)` | Check if Cython compiles, return errors |
| `annotate(code)` | Get annotation score + optimization hints |
| `test(code)` | Run correctness tests against Python reference |
| `benchmark(code)` | Measure speedup vs Python |
| `composite_reward(code, ...)` | Full reward score (compilation + correctness + speed + annotations) |

These are the same tools used during GRPO training — the model learns to use them to iterate on its code, and contributors can use them to optimize their implementations.

## Hardware Instructions in Cython

Using hardware-level operations is explicitly in scope. Cython can access:

- **SIMD via C intrinsics**: `cdef extern from "immintrin.h"` for SSE/AVX
- **OpenMP parallelism**: `from cython.parallel cimport prange` with `nogil`
- **Cache-friendly patterns**: sequential memory access, struct-of-arrays
- **libc functions**: `memcpy`, `memset`, `qsort` — these compile to optimized machine code
- **Math intrinsics**: `libc.math` functions (`sqrt`, `exp`, `log`) use hardware FPU

The goal is to teach the model the full range of optimization techniques available in Cython, from basic type declarations to hardware-level optimization.
