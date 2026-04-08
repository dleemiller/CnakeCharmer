# Common Pitfalls

Mistakes that appear frequently in traces.

## Overview

These are the most common errors observed across ~2,800 agent traces attempting
Cython optimization. Each entry includes the symptom, root cause, and fix.

## Trace Statistics

Aggregate error counts across 6 categories (~2,800 traces):

| Error Type | Count | Fix Rate |
|-----------|-------|----------|
| compilation_error | ~1,935 | 74% |
| test_failure | ~768 | — |
| cdef_placement | ~379 | — |
| segfault | ~245 | — |
| gil_violation | ~152 | 75% |
| timeout | 12 | 60% |

~53% of all traces encountered at least one error requiring iteration.

## Integer Overflow

Python integers have arbitrary precision; C integers overflow silently.

**Symptom**: Tests pass for small inputs, fail for large inputs with wrong values.

```cython
# BAD — overflows for large i
cdef int h = i * 2654435761

# GOOD — cast to prevent overflow
cdef long long h = <long long>i * <long long>2654435761
```

## cdef After Executable Code

All `cdef` declarations must come before any executable statements in a function. **379 occurrences** — the most common placement error.

**Symptom**: `Cdef statement not allowed here` compilation error.

```cython
# BAD
def func(int n):
    x = compute()
    cdef int i        # ERROR: cdef after executable code

# GOOD
def func(int n):
    cdef int i
    x = compute()
```

## Python Objects in nogil Blocks

**Symptom**: `Accessing Python global or builtin not allowed without gil` compilation error.

```cython
# BAD
with nogil:
    result = python_function(x)   # ERROR: Python call needs GIL

# GOOD
with nogil:
    result = c_function(x)        # cdef function, no GIL needed
```

See [[parallelism]] for comprehensive GIL violation patterns.

## Uninitialized Pointers

**Symptom**: Segfault (exit code -11) on first run. 245 segfaults across traces.

```cython
# BAD — no NULL check
cdef int *arr = <int *>malloc(n * sizeof(int))
arr[0] = 1  # segfault if malloc returns NULL

# GOOD
cdef int *arr = <int *>malloc(n * sizeof(int))
if not arr:
    raise MemoryError()
```

## Memory Leaks

**Symptom**: ASan reports leaked bytes; clean code should score 1.0 on memory safety.

```cython
# BAD — early return leaks
cdef int *arr = <int *>malloc(n * sizeof(int))
if condition:
    return -1          # LEAK: arr never freed

# GOOD — free before every return path
cdef int *arr = <int *>malloc(n * sizeof(int))
if not arr:
    raise MemoryError()
try:
    # ... use arr ...
    result = compute(arr)
finally:
    free(arr)
return result
```

## Wrong cimport Paths

Agents frequently invent import paths that don't exist. This is especially common for C standard library functions.

```cython
# BAD — these paths don't exist
from libc.stdlib cimport memset       # ERROR: 'libc/stdlib/memset.pxd' not found
from libc.stddef cimport NULL         # ERROR: 'libc/stddef/NULL.pxd' not found
from cpython.object cimport PyLong_FromLong  # ERROR: 'cpython/object/PyLong_FromLong.pxd' not found

# GOOD — correct import paths
from libc.string cimport memset, memcpy, memcmp   # memset is in libc.string
from libc.stdlib cimport malloc, free, calloc      # NULL is implicitly available
from cpython.long cimport PyLong_FromLong          # if you really need it (usually don't)
```

This error appeared 20+ times across algorithm, dynamic_programming, and graph traces. The most common: `memset` is in `libc.string`, not `libc.stdlib`.

## Closures Inside cpdef

```cython
# BAD — cpdef cannot contain closures
cpdef list sort_items(list data):
    def key_fn(x):          # ERROR: closures inside cpdef functions not yet supported
        return x[1]
    return sorted(data, key=key_fn)

# GOOD — use def instead
def sort_items(list data):
    def key_fn(x):
        return x[1]
    return sorted(data, key=key_fn)
```

Appeared 15+ times across compression and cryptography traces.

## Storing Unsafe C Derivative of Temporary Python Reference

Taking a C pointer to a Python object that may be garbage collected.

```cython
# BAD — temporary bytes object may be freed
cdef const unsigned char* data = some_string.encode('utf-8')
# WARNING: Storing unsafe C derivative of temporary Python reference

# GOOD — keep reference alive
encoded = some_string.encode('utf-8')
cdef const unsigned char* data = encoded

# Also BAD with numpy
cdef double* ptr = <double*>np.zeros(10).data   # temporary ndarray!
# GOOD
arr = np.zeros(10)
cdef double* ptr = <double*>arr.data
```

This warning appeared 25+ times across dynamic_programming, algorithms, and numerical traces.

## Cannot Coerce Python List to Memoryview

```cython
# BAD — Python lists aren't buffers
cdef int[::1] arr = [0] * n    # ERROR: Cannot coerce multiplied list to 'int[:]'
cdef int[:] view = [1, 2, 3]   # ERROR

# GOOD — use numpy
import numpy as np
cdef int[::1] arr = np.zeros(n, dtype=np.intc)
```

14+ occurrences in graph traces. See [[memoryviews]] for more.

## Casting Python Objects to C Pointers

```cython
# BAD — Python object to primitive pointer
cdef int* ptr = <int*>python_list
# ERROR: Casting temporary Python object to non-numeric non-Python type
# ERROR: Python objects cannot be cast to pointers of primitive types

# GOOD — go through numpy or malloc
import numpy as np
arr = np.array(python_list, dtype=np.intc)
cdef int[::1] view = arr
```

6+ occurrences in dynamic_programming traces.

## Variable Redeclaration

```cython
# BAD — declaring same variable twice
def func(int n):
    cdef int idx
    cdef int idx        # ERROR: 'idx' redeclared

# GOOD — declare once
def func(int n):
    cdef int idx
```

## Special Methods with cdef

```cython
# BAD — cdef for special methods
cdef class Node:
    cdef int __len__(self):   # ERROR: Special methods must be declared with 'def', not 'cdef'
        return self.size

# GOOD
cdef class Node:
    def __len__(self):
        return self.size
```

See [[extension-types]] for more on cdef classes.

## Undeclared malloc/free

```cython
# BAD — using malloc without cimport
cdef int* arr = <int*>malloc(n * sizeof(int))
# ERROR: undeclared name not builtin: malloc

# GOOD — cimport first
from libc.stdlib cimport malloc, free
cdef int* arr = <int*>malloc(n * sizeof(int))
```

Appeared 20+ times across all categories. See [[memory-management]].

## Invalid prange Keywords

```cython
# BAD — 'reduction' is not a valid prange keyword
for i in prange(n, nogil=True, reduction='+'):
    total += arr[i]
# ERROR: Invalid keyword argument: reduction

# GOOD — prange auto-detects reductions from +=, *=, etc.
for i in prange(n, nogil=True):
    total += arr[i]    # auto-reduced
```

22 occurrences in numerical traces. See [[parallelism]].

## Unrecognized Characters

```cython
# BAD — Unicode or special characters in .pyx files
# Often from copy-pasting from documentation with smart quotes or non-ASCII
# ERROR: Unrecognized character

# GOOD — ensure file is clean ASCII/UTF-8 with no BOM
```

## Not Allowed in Constant Expression

```cython
# BAD — using runtime values in compile-time contexts
DEF SIZE = len(some_list)    # ERROR: Not allowed in a constant expression

# GOOD — use literal constants
DEF SIZE = 256
```

## See Also

[[typing]], [[memory-management]], [[c-interop]], [[parallelism]], [[error-handling]]
