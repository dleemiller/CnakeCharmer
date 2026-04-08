# NumPy Interop

Typed NumPy arrays, ufuncs, and Pythran integration.

## Overview

NumPy arrays accessed through typed memoryviews give C-speed element access.
`@cython.ufunc` creates NumPy universal functions from scalar Cython code.
Pythran can fuse NumPy expression trees for vectorized operations.

## Syntax

```cython
# Typed memoryview from NumPy array
import numpy as np
cimport numpy as cnp

def ewma(double[::1] data, double alpha):
    cdef int i, n = data.shape[0]
    cdef double[::1] out = np.empty(n)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i-1]
    return np.asarray(out)

# Typed with cnp types
def histogram(cnp.float64_t[::1] data, int n_bins):
    ...

# @cython.ufunc — auto-broadcasts, auto-parallelizes
@cython.ufunc
cdef double smoothstep(double x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return x * x * (3.0 - 2.0 * x)

# Fused type ufunc
@cython.ufunc
cdef cython.floating sigmoid(cython.floating x):
    return 1.0 / (1.0 + exp(-x))
```

## The #1 Error: 'np' is not a cimported module

This is the **most common NumPy-related compilation error** across all traces — 67+ occurrences in numerical alone. It happens when you use `import numpy as np` but try to use NumPy features that require `cimport`.

```cython
# BAD — only import, no cimport
import numpy as np
def func(np.float64_t[::1] data):    # ERROR: 'np' is not a cimported module
    ...

# GOOD — both import AND cimport
import numpy as np
cimport numpy as cnp

def func(cnp.float64_t[::1] data):   # use cnp (cimported) for types
    cdef cnp.ndarray[double, ndim=1] arr = np.empty(10)  # np for allocation
    ...
```

**Rule**: Use `import numpy as np` for runtime operations (allocation, math). Use `cimport numpy as cnp` for compile-time type declarations. Many traces use `np` for both — that's the bug.

Similarly for arrays:

```cython
# BAD
from array import array
cdef array.array a = array('i', [1,2,3])   # ERROR: 'array' is not a cimported module

# GOOD
from cpython cimport array
import array
cdef array.array a = array.array('i', [1,2,3])
```

## Don't Fight NumPy SIMD

NumPy operations like `np.sum()`, `np.dot()`, `a + b` are already SIMD-optimized in C. Don't replace them with scalar Cython loops unless you have a specific reason.

**When Cython beats NumPy**:
- Fused operations (combine multiple passes into one loop)
- Custom reductions not available in NumPy
- Element-wise operations with branching/conditionals
- Operations that need state between elements (running totals, EWMA)
- Avoiding temporary array allocation

**When NumPy beats Cython**:
- Simple vectorized math (`a + b`, `np.sqrt(a)`)
- Matrix multiplication (`np.dot`, `@` operator)
- Operations on very large arrays (NumPy's BLAS/LAPACK)

```cython
# BAD — reimplementing what NumPy does better
def slow_add(double[::1] a, double[::1] b):
    cdef int i, n = a.shape[0]
    cdef double[::1] out = np.empty(n)
    for i in range(n):
        out[i] = a[i] + b[i]    # NumPy's np.add is SIMD-optimized
    return np.asarray(out)

# GOOD — fuse operations NumPy can't
def fused_normalize(double[::1] data):
    """Compute mean, subtract it, and divide by std in ONE pass."""
    cdef int i, n = data.shape[0]
    cdef double total = 0.0, sq_total = 0.0, mean, std
    cdef double[::1] out = np.empty(n)

    for i in range(n):
        total += data[i]
    mean = total / n

    for i in range(n):
        out[i] = data[i] - mean
        sq_total += out[i] * out[i]
    std = (sq_total / n) ** 0.5

    if std > 0:
        for i in range(n):
            out[i] /= std
    return np.asarray(out)
```

## Creating Arrays Inside Loops

A common performance trap: allocating NumPy arrays in a hot loop.

```cython
# BAD — allocation per iteration
def bad_process(double[:, ::1] matrix):
    cdef int i, rows = matrix.shape[0]
    for i in range(rows):
        row = np.array(matrix[i, :])   # allocation on every iteration!
        ...

# GOOD — allocate once, reuse
def good_process(double[:, ::1] matrix):
    cdef int i, rows = matrix.shape[0], cols = matrix.shape[1]
    cdef double[::1] row_buf = np.empty(cols)
    for i in range(rows):
        row_buf[:] = matrix[i, :]      # copy into pre-allocated buffer
        ...
```

## dtype Matching

Ensure numpy dtype matches the Cython memoryview type:

```cython
# BAD — Python float (float64) to int memoryview
cdef int[::1] arr = np.zeros(n)        # np.zeros defaults to float64!

# GOOD — explicit dtype
cdef int[::1] arr = np.zeros(n, dtype=np.intc)          # int
cdef long long[::1] arr = np.zeros(n, dtype=np.int64)   # long long
cdef double[::1] arr = np.zeros(n, dtype=np.float64)    # double (np.float64)
cdef float[::1] arr = np.zeros(n, dtype=np.float32)     # float
```

## Trace Statistics

Across ~2,800 traces:

| Error Pattern | Count | Category |
|--------------|-------|----------|
| 'np' is not a cimported module | 67+ | numerical (dominant), graph, dp |
| 'array' is not a cimported module | 11+ | dynamic_programming, graph |
| dtype mismatch errors | scattered | all categories |

The `cimport` error alone accounts for a significant fraction of all compilation errors in numerical problems.

## Gotchas

1. **cimport AND import** — You need both: `import numpy as np` (runtime) + `cimport numpy as cnp` (compile-time types).
2. **Use cnp for types** — `cnp.float64_t`, `cnp.ndarray`, not `np.float64_t`.
3. **Default dtype** — `np.zeros(n)` is float64, not int. Always specify `dtype=`.
4. **Don't loop over numpy ops** — `np.sum()` is faster than a Cython loop over elements for simple reductions.
5. **Fuse for wins** — Combine multiple numpy operations into one Cython loop to avoid temporary arrays.
6. **Return as numpy** — Use `np.asarray(memoryview)` to return results to Python.

## See Also

[[memoryviews]], [[typing]], [[optimization]], [[c-interop]]
