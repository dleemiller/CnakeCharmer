# Optimization

Maximizing speedup and annotation quality.

## Overview

Cython generates HTML annotations showing which lines involve Python/C API calls
(yellow) vs pure C (white). The annotation score (white/total ratio) is our
primary optimization metric alongside wall-clock speedup.

## Annotation Score

```
score = white_lines / total_lines
```

- **> 0.9**: Excellent — most code is pure C
- **0.7–0.9**: Good — some Python interaction remains
- **< 0.7**: Needs work — too many Python fallback lines

Common causes of yellow lines:
- Python object creation in loops
- Untyped variables (falls back to `PyObject *`)
- Python function calls (use `cdef` or `cpdef` instead)
- String/dict/list operations

### Annotation Scores from Traces

Across ~2,800 traces:

| Category | Avg Annotation | Avg Speedup | Notes |
|----------|---------------|-------------|-------|
| algorithms | 0.867 | 782x | Diverse patterns |
| numerical | 0.858 | 1,427x | Heavy numpy interaction |
| dynamic_programming | 0.898 | 6,464x | Clean loop structures |
| graph | 0.846 | 4,750x | Pointer-heavy, more complex |
| cryptography | 0.876 | 2,211x | Bit manipulation dominant |
| compression | 0.891 | 5,450x | Byte-level processing |

**Key insight**: Dynamic programming achieves the highest speedups (6,464x avg) and annotation scores (0.898) because DP problems have clean nested loops that map perfectly to typed C code.

## Optimization Checklist

1. **Type everything** — all loop variables, function params, local accumulators
2. **C math** — `from libc.math cimport` instead of Python `math` module
3. **C arrays** — `malloc`/`free` instead of Python lists in hot loops
4. **Avoid Python objects** — no dicts, lists, sets inside inner loops
5. **Use `cdef` functions** — for internal helper functions called in loops
6. **Precompute** — move invariant calculations outside loops
7. **Release the GIL** — `with nogil:` if no Python objects are needed
8. **Directives** — `boundscheck=False, wraparound=False` for array-heavy code

## Patterns

### The Biggest Win: Typing Loop Variables

The single most impactful optimization. An untyped loop variable forces Python integer objects on every iteration:

```cython
# BAD — i is a Python object, ~1x speedup
def slow_sum(data):
    total = 0.0
    for i in range(len(data)):
        total += data[i]
    return total

# GOOD — typed everything, annotation score jumps from ~0.3 to ~0.9
def fast_sum(double[::1] data):
    cdef int i, n = data.shape[0]
    cdef double total = 0.0
    for i in range(n):
        total += data[i]
    return total
```

### malloc vs Memoryview vs NumPy

Choose based on context:

```cython
# malloc — best for: temporary scratch space, C-interop, nogil blocks
# Pro: no Python overhead, works in nogil
# Con: must manually free, no bounds checking
from libc.stdlib cimport malloc, free

cdef int* scratch = <int*>malloc(n * sizeof(int))
if not scratch:
    raise MemoryError()
# ... use scratch ...
free(scratch)

# Memoryview — best for: function parameters, passing arrays between functions
# Pro: typed, bounds-checked (optional), slice-friendly
# Con: slight overhead vs raw pointer
def process(double[::1] data):
    cdef int i
    for i in range(data.shape[0]):
        data[i] *= 2.0

# NumPy — best for: return values, allocation, vectorized operations
# Pro: Python-visible, garbage collected, SIMD-optimized operations
# Con: overhead for element-wise access in loops
import numpy as np
cdef double[::1] result = np.empty(n)
```

### Timeout Prevention

12 timeouts observed across traces (60% fix rate). Common causes:

```cython
# BAD — infinite loop from integer overflow
cdef int i = 0
while i < n:
    # if n > INT_MAX, i wraps around and never reaches n
    i += 1

# GOOD — use appropriate integer size
cdef long long i = 0
while i < n:
    i += 1

# BAD — O(n^3) algorithm with no early exit
for i in range(n):
    for j in range(n):
        for k in range(n):
            ...

# GOOD — algorithmic optimization first, then Cython
# Consider: can you reduce complexity before micro-optimizing?
```

### cdef Helper Functions

Extract hot inner operations to cdef functions for inlining:

```cython
# The cdef function can be inlined by the C compiler
cdef inline double squared_distance(double x1, double y1, double x2, double y2) noexcept nogil:
    cdef double dx = x1 - x2
    cdef double dy = y1 - y2
    return dx * dx + dy * dy

def nearest_neighbor(double[::1] xs, double[::1] ys, double qx, double qy):
    cdef int i, best_i = 0
    cdef double d, best_d = squared_distance(xs[0], ys[0], qx, qy)
    for i in range(1, xs.shape[0]):
        d = squared_distance(xs[i], ys[i], qx, qy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i
```

### C Math Library

Replace Python math with C math for nogil compatibility and speed:

```cython
# BAD — Python math requires GIL
import math
result = math.sqrt(x)     # Python call → yellow line

# GOOD — C math is GIL-free
from libc.math cimport sqrt, exp, log, sin, cos, fabs, pow, floor, ceil
result = sqrt(x)           # C call → white line
```

## Speedup Distribution

From traces, speedup ranges are enormous:

- **10–100x**: Typical for simple loop typing without algorithmic changes
- **100–1,000x**: Good typing + directives + cdef helpers
- **1,000–10,000x**: Full optimization with malloc, nogil, clean C loops
- **10,000x+**: Problems where Python is pathologically slow (e.g., deep recursion, element-wise ops on large arrays)

Most speedup comes from **typing**, not from parallelism. A well-typed single-threaded Cython function typically beats an untyped parallel one.

## Trace Statistics

Across ~2,800 traces from 6 categories:

| Metric | Value |
|--------|-------|
| Overall avg annotation | 0.873 |
| Overall avg speedup | ~3,500x |
| Traces with errors | ~53% |
| Timeout count | 12 |
| Timeout fix rate | 60% |

## Gotchas

1. **Type before parallelize** — Typing gives 100-1000x; prange gives 2-8x on top. Do typing first.
2. **Annotation score != speedup** — A 0.95 score with bad algorithmic complexity is still slow.
3. **NumPy in loops** — `np.zeros()` inside a loop allocates on every iteration. Allocate once outside.
4. **Python math** — `math.sqrt` is a Python call. Use `libc.math.sqrt` for white lines.
5. **Integer overflow** — C `int` is 32-bit. Use `long long` for large values to avoid silent overflow.
6. **Timeouts** — Usually from infinite loops caused by integer overflow, not from slow algorithms.

## See Also

[[compiler-directives]], [[typing]], [[parallelism]], [[memoryviews]]
