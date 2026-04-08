# Parallelism

`prange` and `nogil` for multi-threaded Cython.

## Overview

Cython can release the GIL and run loops in parallel via OpenMP. `prange`
replaces `range` in for loops, distributing iterations across threads.
Only C-typed operations (no Python objects) are allowed in `nogil` blocks.

## Syntax

```cython
from cython.parallel cimport prange, parallel

# Basic prange with reduction
def sum_squares(int n):
    cdef double total = 0.0
    cdef int i
    for i in prange(n, nogil=True):
        total += <double>i * <double>i    # += is auto-reduced
    return total

# Schedule control
for i in prange(n, nogil=True, schedule='static'):    # equal chunks (default)
    ...
for i in prange(n, nogil=True, schedule='dynamic'):   # work-stealing (uneven work)
    ...

# nogil block (no Python objects allowed)
with nogil:
    for i in range(n):
        c_function(data[i])

# Parallel block with thread-local variables
with nogil, parallel():
    tid = cython.parallel.threadid()
```

## GIL Violation Errors

GIL violations are the **#1 parallelism error** across all trace categories. Across ~2,800 traces, 163 GIL/nogil violations were recorded.

### Common Error Messages

All of these mean "you're using Python objects inside a nogil block":

| Error Message | Occurrences | Fix Rate |
|--------------|-------------|----------|
| "Calling gil-requiring function not allowed without gil" | ~40 | 84% |
| "Coercion from Python not allowed without the GIL" | ~30 | 84% |
| "Converting to Python object not allowed without gil" | ~20 | 84% |
| "Constructing Python tuple not allowed without gil" | ~20 | 45% |
| "Operation not allowed without gil" | ~15 | 84% |
| "Indexing Python object not allowed without gil" | ~5 | 95% |
| "Assignment of Python object not allowed without gil" | ~5 | 95% |

### Pattern: Python Call in nogil

```cython
# BAD — calling Python function without GIL
with nogil:
    result = python_function(x)   # ERROR: Calling gil-requiring function

# GOOD — use a cdef function
cdef double c_function(double x) noexcept nogil:
    return x * x

with nogil:
    result = c_function(x)
```

### Pattern: Constructing Python Tuples in nogil

This is especially common in numerical code (45% fix rate — hardest GIL error to resolve):

```cython
# BAD — tuples are Python objects
with nogil:
    pair = (x, y)     # ERROR: Constructing Python tuple not allowed without gil

# GOOD — use C struct or separate variables
cdef struct Point:
    double x
    double y

with nogil:
    cdef Point p
    p.x = x
    p.y = y
```

### Pattern: Python Coercion in prange

```cython
# BAD — Python list indexing in prange
for i in prange(n, nogil=True):
    val = my_list[i]      # ERROR: Coercion from Python not allowed without the GIL

# GOOD — use typed memoryview
cdef double[::1] arr = np.array(my_list, dtype=np.float64)
for i in prange(n, nogil=True):
    val = arr[i]          # typed memoryview — no GIL needed
```

## nogil Helper Functions

The `noexcept nogil` pattern is critical for helpers called in nogil blocks:

```cython
# BAD — except * (default) requires GIL for exception checking
cdef int compare(const void *a, const void *b):    # implicit except *
    ...
# PERFORMANCE HINT: "Exception check on 'compare' will always require the GIL to be acquired"

# GOOD — noexcept tells Cython no exception checking needed
cdef int compare(const void *a, const void *b) noexcept nogil:
    cdef int va = (<int *>a)[0]
    cdef int vb = (<int *>b)[0]
    return va - vb
```

## qsort Callback Type Signatures

A frequent graph/algorithms error — qsort expects `int (*)(const void *, const void *)` with no exception clause:

```cython
from libc.stdlib cimport qsort

# BAD — except * is incompatible with C function pointer
cdef int cmp(const void *a, const void *b) except *:   # implicit default
    ...
qsort(arr, n, sizeof(int), cmp)
# ERROR: "Cannot assign type 'int (*)(const void *, const void *) except *' to 'int (*)(const void *, const void *)'"

# BAD — except? -1 also incompatible
cdef int cmp(const void *a, const void *b) except? -1 nogil:
    ...
# ERROR: "Cannot assign type 'int (...) except? -1 nogil' to 'int (*)(...)"

# GOOD — noexcept matches C function pointer type
cdef int cmp(const void *a, const void *b) noexcept nogil:
    return (<int *>a)[0] - (<int *>b)[0]

qsort(arr, n, sizeof(int), cmp)   # works
```

This error appeared 12+ times in graph and algorithm traces.

## When NOT to Use prange

**Do not use prange when the Python baseline is single-threaded.** Our benchmarks compare Cython against single-threaded Python. Using prange gives an unfair multi-core advantage that doesn't reflect real Cython optimization skill.

prange is appropriate when:
- The Python baseline also uses multiprocessing/threading
- The workload is genuinely CPU-bound and embarrassingly parallel
- Each iteration is independent (no data dependencies)

prange is inappropriate when:
- The loop body has data dependencies between iterations
- The workload is memory-bound (prange adds overhead, no speedup)
- Iterations are very short (thread overhead dominates)

## Fix Rates by Category

| Category | GIL Violations | Fix Rate | Notes |
|----------|---------------|----------|-------|
| algorithms | 44 | 84% | Mostly function calls in nogil |
| numerical | 33 | 45% | Tuple construction is hard to fix |
| dynamic_programming | 14 | 91% | Straightforward loop patterns |
| graph | 21 | 95% | qsort callbacks need noexcept |
| cryptography | 42 | 51% | Complex bit operations need careful typing |
| compression | 9 | 100% | Simple patterns, easy fixes |

## Gotchas

1. **No Python objects in nogil** — Strings, lists, dicts, tuples, and Python function calls all require the GIL.
2. **prange auto-reduction** — `+=`, `-=`, `*=` are automatically reduced. Don't pass `reduction=` keyword (it doesn't exist).
3. **noexcept for callbacks** — C function pointers (qsort, bsearch) need `noexcept` — the default `except *` is incompatible.
4. **Performance hints** — "Exception check will always require the GIL" means add `noexcept` to avoid hidden GIL acquisition.
5. **Thread safety** — Shared mutable state in prange causes data races. Use thread-local variables or atomic operations.
6. **Single-threaded baselines** — Don't use prange just for benchmark numbers when Python baseline is single-threaded.

## See Also

[[optimization]], [[typing]], [[error-handling]], [[compiler-directives]]
