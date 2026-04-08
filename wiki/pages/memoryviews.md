# Typed Memoryviews

Fast, GIL-free access to contiguous memory buffers.

## Overview

Memoryviews provide C-pointer-speed access to NumPy arrays, C arrays, and any
buffer-protocol object. They support slicing, transposition, and can be passed
to C functions via `&view[0]`. The key to performance is specifying contiguity.

## Syntax

```cython
# 1D, C-contiguous (fastest for row access)
def process(double[::1] data):
    cdef int i, n = data.shape[0]
    for i in range(n):
        data[i] *= 2.0

# 2D, C-contiguous
def matrix_op(double[:, ::1] mat):
    cdef int rows = mat.shape[0], cols = mat.shape[1]

# 2D, Fortran-contiguous (column-major)
def col_sum(double[::1, :] mat):
    ...

# Const (read-only, allows compiler optimizations)
def dot(const double[::1] a, const double[::1] b):
    cdef double s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s

# From malloc'd pointer
cdef double *buf = <double *>malloc(n * sizeof(double))
cdef double[:] view = <double[:n]>buf

# Allocate via cython.view.array
from cython.view cimport array
cdef array tmp = array(shape=(rows, cols), itemsize=sizeof(double), format='d')
cdef double[:, ::1] mat = tmp

# Operations
cdef double[:] copy = view.copy()   # deep copy
cdef double[:, :] t = mat.T         # transpose (swaps strides)
```

## Patterns

### Cannot Convert C Pointer to Memoryview

You cannot directly assign a C pointer to a typed memoryview — you need the cast syntax:

```cython
from libc.stdlib cimport malloc, free

# BAD — direct assignment fails
cdef int* buf = <int*>malloc(n * sizeof(int))
cdef int[::1] view = buf       # ERROR: Cannot convert 'int *' to memoryviewslice

# GOOD — use cast syntax
cdef int* buf = <int*>malloc(n * sizeof(int))
cdef int[::1] view = <int[:n]>buf   # cast pointer to 1D memoryview

# For 2D:
cdef double* buf2d = <double*>malloc(rows * cols * sizeof(double))
cdef double[:, ::1] mat = <double[:rows, :cols]>buf2d
```

This error appeared 8+ times in dynamic_programming and graph traces.

### Cannot Convert void* to Memoryview

malloc returns `void*` — you must cast to the typed pointer first:

```cython
# BAD — void* directly to memoryview
cdef int[::1] view = <int[:n]>malloc(n * sizeof(int))
# ERROR: Cannot convert 'void *' to memoryviewslice

# GOOD — cast to typed pointer first
cdef int* buf = <int*>malloc(n * sizeof(int))
cdef int[::1] view = <int[:n]>buf
```

### Memoryview Shape Conformability

Dimensions must match exactly:

```cython
# BAD — 1D memoryview assigned to 2D parameter
cdef int[::1] flat = np.zeros(n, dtype=np.intc)
cdef int[:, ::1] mat = flat    # ERROR: Memoryview 'int[::1]' not conformable to memoryview 'int[:, ::1]'

# GOOD — reshape first
import numpy as np
cdef int[:, ::1] mat = np.zeros((rows, cols), dtype=np.intc)

# Also BAD — wrong contiguity
cdef int[:] generic = np.zeros(n, dtype=np.intc)
cdef int[::1] contiguous = generic  # ERROR: 'int[:]' not conformable to 'int[::1]'

# GOOD — ensure contiguity at creation
cdef int[::1] contiguous = np.ascontiguousarray(data, dtype=np.intc)
```

### Cannot Take Address of Memoryview Slice

```cython
# BAD — slicing creates a temporary
cdef int* ptr = &view[1:5]     # ERROR: Cannot take address of memoryview slice

# GOOD — take address of first element of the slice
cdef int* ptr = &view[1]       # pointer to element at index 1
# Then use ptr[0], ptr[1], ..., ptr[3] for elements 1-4
```

### Python List to Memoryview

Python lists cannot be directly assigned to memoryviews:

```cython
# BAD — Python list to typed memoryview
cdef int[::1] arr = [0] * n    # ERROR: Cannot coerce multiplied list to 'int[:]'

# GOOD — create numpy array first
import numpy as np
cdef int[::1] arr = np.zeros(n, dtype=np.intc)

# GOOD — for small fixed arrays, use C array
cdef int arr[256]    # stack-allocated, no Python overhead
```

This error appeared 14+ times in graph traces. It's one of the most common memoryview mistakes.

### Memoryview Index Type Errors

```cython
# BAD — using wrong type to index
cdef int[:] view = ...
cdef long val = 5
view[val] = 10    # may warn depending on context

# GOOD — use Py_ssize_t or int for indices
cdef Py_ssize_t idx = 5
view[idx] = 10
```

## Returning Memoryviews

Memoryviews cannot be directly returned to Python. Wrap in numpy:

```cython
import numpy as np

def compute(int n):
    cdef double[::1] result = np.empty(n)
    cdef int i
    for i in range(n):
        result[i] = <double>i * <double>i
    return np.asarray(result)    # convert back to numpy for Python
```

## Trace Statistics

Across ~2,800 traces from 6 categories:

| Error Pattern | Count | Categories |
|--------------|-------|------------|
| Cannot convert pointer to memoryviewslice | 8+ | dynamic_programming, graph |
| Memoryview not conformable (shape mismatch) | 4+ | dynamic_programming, graph |
| Cannot coerce multiplied list to memoryview | 14+ | graph |
| Cannot convert void* to memoryviewslice | 3+ | dynamic_programming |
| Cannot take address of memoryview slice | 1+ | dynamic_programming |
| Invalid index for memoryview | 2+ | graph |

## Gotchas

1. **Lists aren't memoryviews** — `[0] * n` is a Python list. Use `np.zeros(n, dtype=...)`.
2. **Cast pointers properly** — `malloc` → cast to typed pointer → cast to memoryview via `<type[:size]>ptr`.
3. **Shape must match** — 1D can't go into 2D. Contiguous (`::1`) can't go into generic (`:`) slots easily.
4. **Return as numpy** — Use `np.asarray(memoryview)` to return memoryviews to Python callers.
5. **Contiguity matters** — `double[::1]` (C-contiguous) is faster than `double[:]` (strided). Always specify when possible.

## See Also

[[numpy-interop]], [[memory-management]], [[parallelism]], [[typing]]
