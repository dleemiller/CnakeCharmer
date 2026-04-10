# Contributing

## Dataset Structure

```
cnake_data/
  py/{category}/{name}.py       # Python implementation (training prompt)
  cy/{category}/{name}.pyx      # Cython implementation (ground truth)
tests/
  data/{category}/test_{name}.py  # Equivalence tests
```

Categories: `algorithms`, `numerical`, `sorting`, `string_processing`, `math_problems`, `dynamic_programming`, `geometry`, `simulation`, `graph`, `statistics`, `cryptography`, `nn_ops`, `image_processing`, `compression`, `leetcode`, `physics`, `diff_equations`, `dsp`, `optimization`

New category directories need `__init__.py` in `cnake_data/py/`, `cnake_data/cy/`, and `tests/data/`.

## Adding a Problem

### 1. Python implementation

Create `cnake_data/py/{category}/{name}.py`:

```python
"""Brief description.

Keywords: relevant, search, terms
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))  # should take ~10-500ms
def func_name(n: int) -> return_type:
    # Pure Python, no external deps, deterministic
    ...
```

**Signature diversity matters.** This is training data — don't make every function take a single `n: int`. Use realistic patterns: multiple params, pre-built data structures, string inputs, class-based designs.

### 2. Cython implementation

Create `cnake_data/cy/{category}/{name}.pyx`:

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def func_name(int n):
    cdef int i
    cdef double *arr = <double *>malloc(n * sizeof(double))
    if not arr:
        raise MemoryError()
    # C-optimized implementation ...
    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
```

**Key optimization techniques:**
- `cdef` all loop variables and accumulators
- C arrays (`malloc`/`free`) instead of Python lists in hot loops
- `from libc.math cimport sqrt, exp, log` instead of Python `math`
- `with nogil:` around pure-C loop sections
- Convert to Python objects only when returning

**Do NOT use:** `PyList_SET_ITEM`/`PyList_New` (segfaults), `import cython` annotation syntax in `.pyx` files, `list.append()` in hot loops.

**When the Python baseline uses NumPy:** don't replace vectorized NumPy with scalar Cython loops. Instead, fuse multiple NumPy ops to eliminate temporaries, write custom reductions NumPy can't express, or reduce Python overhead between NumPy calls.

### 3. Choose a discriminating return value

Return values verify py/cy equivalence. A good return value **changes when the algorithm is wrong**.

- Return full data structures when n is small enough
- For large outputs, return a tuple of 2-4 independent indicators: `(total, dp[n//2], max_val)`
- Never return a bare count or sum alone — pair with a sample or checksum

### 4. Write the equivalence test

Create `tests/data/{category}/test_{name}.py`:

```python
import pytest
from cnake_data.py.{category}.{name} import func_name as py_func
from cnake_data.cy.{category}.{name} import func_name as cy_func

@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_{name}_equivalence(n):
    assert py_func(n) == cy_func(n)
```

For floats: `assert abs(py - cy) / max(abs(py), 1.0) < 1e-4`

### 5. Compile, test, score

```bash
make compile
uv run --no-sync pytest tests/data/{category}/test_{name}.py -v
```

Score with MCP: `score_problem("{category}/{name}")`

**Acceptance thresholds:**

| Signal | Minimum | Target |
|--------|---------|--------|
| Total reward | >= 0.80 | > 0.90 |
| Correctness | 1.0 (required) | -- |
| Speedup | >= 5x | > 10x |
| Annotation | >= 0.85 | > 0.90 |
| Memory safety | 1.0 (required) | -- |
| Lint | 1.0 (required) | -- |

Review the HTML annotation (`cnake_data/cy/{category}/{name}.html`): yellow lines = Python (slow), white lines = C (fast). Target >85% C lines.

If speedup is low, check whether the Python baseline uses C-level builtins (`list.sort()`, `sum()`) — replace with pure Python loops so the speedup reflects real Cython value.

### 6. Run benchmarks and commit

```bash
make benchmark
git add cnake_data/py/ cnake_data/cy/ tests/data/ docs/BENCHMARKS.md .benchmark_cache.json
```

## No Templates / No Clones

Every problem must be algorithmically distinct. Do not mass-produce problems by copying implementations and renaming symbols. If two problems share more than superficial similarity, keep one and delete the others.

## Memory Safety (ASan)

The pipeline includes AddressSanitizer checking (15% of composite reward). It detects leaks, buffer overflows, use-after-free, and double-free by compiling with `-fsanitize=address` and running with small inputs.

```python
from cnake_charmer.eval.memory_safety import check_memory_safety
code = open("cnake_data/cy/algorithms/max_flow.pyx").read()
result = check_memory_safety(code, "max_flow", test_args=(100,))
print(result.score, result.errors)
```

## Reward Weights

| Signal | Weight |
|--------|--------|
| Correctness | 30% |
| Performance (speedup) | 25% |
| Annotations (C-level code) | 20% |
| Memory safety (ASan) | 15% |
| Lint (cython-lint) | 10% |

## MCP Tools

```bash
claude mcp add cnake-charmer -- uv run python -m cnake_charmer.mcp_server
```

| Tool | Description |
|------|-------------|
| `score_problem` | Full composite reward (compile + test + benchmark + annotate + ASan) |
| `list_problems` | List all problem pairs with status |
| `compile_file` | Compile a .pyx and check for errors |
| `annotate_file` | HTML annotations for optimization analysis |
| `check_memory` | AddressSanitizer memory error detection |
| `evaluate_cython` | Compile + test + benchmark from raw code strings |

## Finding New Problems

Primary source: **The Stack v2 (dedup, Cython subset)** in `scripts/utils/stack_data/stack_cython_full.duckdb`.

The `split` column tracks conversion status: `sft` (done), `sft_candidate` (selected, not yet processed), `grpo_candidate` (for GRPO training), `NULL` (unpartitioned).

```sql
-- Query candidates
SELECT blob_id, filename, content, length_bytes
FROM stack_cython
WHERE split = 'sft_candidate'
ORDER BY length_bytes ASC LIMIT 30;

-- Mark converted
UPDATE stack_cython SET split = 'sft' WHERE blob_id = '<blob_id>';
```

**Adaptation workflow:** Read source -> write pure Python version -> write Cython version -> score with MCP -> iterate until >= 0.80 reward -> update DuckDB split.
