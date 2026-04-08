# Memory Management

Manual memory allocation: malloc, free, calloc, realloc, buffer protocol.

## Overview

For hot loops that can't use Python objects, Cython provides direct access to
C memory allocation. Always pair `malloc` with `free`, check for NULL returns,
and use `__dealloc__` in `cdef class` for RAII-style cleanup.

## Trace Statistics

| Source | Traces | Segfaults | malloc/free Undeclared | libc Path Errors |
|--------|--------|-----------|----------------------|-----------------|
| algorithms | 500 | 53 | 10+ | 4 |
| graph | 500 | 137 | 10+ | 6 |
| cryptography | 408 | 3 | -- | -- |
| **Total** | **1408** | **193** | **~20** | **~10** |

Top memory-related compile errors across 1408 traces:
- `undeclared name not builtin: malloc` -- ~10 instances
- `undeclared name not builtin: free` -- ~10 instances
- `'libc/stdlib/memset.pxd' not found` -- 10 instances (wrong cimport path; see [[c-interop]])
- `Non-extern C function declared but not defined` -- 1 instance

## Syntax

```cython
from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset, memcmp

# Basic malloc pattern
cdef int *arr = <int *>malloc(n * sizeof(int))
if not arr:
    raise MemoryError()
try:
    # ... use arr ...
    pass
finally:
    free(arr)

# calloc -- zero-initialized
cdef int *counts = <int *>calloc(256, sizeof(int))

# realloc -- resize
cdef int *new_arr = <int *>realloc(arr, new_size * sizeof(int))
if not new_arr:
    free(arr)
    raise MemoryError()
arr = new_arr

# memcpy, memset
memcpy(dst, src, n * sizeof(double))
memset(arr, 0, n * sizeof(int))

# RAII via cdef class
cdef class DynamicArray:
    cdef int *data
    cdef int size, capacity

    def __cinit__(self, int capacity):
        self.data = <int *>malloc(capacity * sizeof(int))
        if not self.data:
            raise MemoryError()
        self.size = 0
        self.capacity = capacity

    def __dealloc__(self):
        if self.data:
            free(self.data)

# Buffer protocol (expose C memory to Python/NumPy)
cdef class DoubleBuffer:
    cdef double *data
    cdef Py_ssize_t n

    def __getbuffer__(self, Py_buffer *buf, int flags):
        buf.buf = <void *>self.data
        buf.len = self.n * sizeof(double)
        buf.itemsize = sizeof(double)
        buf.ndim = 1
        # ... set shape, strides, format ...

    def __releasebuffer__(self, Py_buffer *buf):
        pass
```

## Patterns

### The cimport requirement

The most common memory-management compile error is using `malloc` or `free`
without the proper cimport. Cython does not provide these as builtins -- you
must explicitly import them from `libc.stdlib`.

**BAD** -- missing cimport causes `undeclared name not builtin: malloc`:
```cython
# No cimport!
def allocate(int n):
    cdef int *arr = <int *>malloc(n * sizeof(int))  # ERROR
    free(arr)                                         # ERROR
```

**GOOD** -- proper cimport at module level:
```cython
from libc.stdlib cimport malloc, free

def allocate(int n):
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()
    try:
        pass  # use arr
    finally:
        free(arr)
```

### NULL pointer checks after every allocation

Every `malloc`, `calloc`, or `realloc` can return NULL. Skipping the check
leads to segfaults -- the #1 runtime error in graph traces (137 segfaults out
of 500 traces).

**BAD** -- no NULL check:
```cython
from libc.stdlib cimport malloc, free

cdef int *data = <int *>malloc(n * sizeof(int))
data[0] = 42  # segfault if malloc returned NULL
```

**GOOD** -- always check:
```cython
from libc.stdlib cimport malloc, free

cdef int *data = <int *>malloc(n * sizeof(int))
if not data:
    raise MemoryError()
data[0] = 42
```

### try/finally for cleanup

The biggest source of memory leaks is early returns or exceptions that skip
the `free()` call. Always wrap manual memory in `try/finally`.

**BAD** -- exception path leaks memory:
```cython
from libc.stdlib cimport malloc, free

def process(int n):
    cdef int *buf = <int *>malloc(n * sizeof(int))
    if not buf:
        raise MemoryError()
    # If this raises, buf is never freed!
    result = do_work(buf, n)
    free(buf)
    return result
```

**GOOD** -- try/finally guarantees cleanup:
```cython
from libc.stdlib cimport malloc, free

def process(int n):
    cdef int *buf = <int *>malloc(n * sizeof(int))
    if not buf:
        raise MemoryError()
    try:
        result = do_work(buf, n)
        return result
    finally:
        free(buf)
```

### Multiple allocations with cascading cleanup

When allocating multiple buffers, free everything on failure.

**BAD** -- second malloc fails, first buffer leaks:
```cython
from libc.stdlib cimport malloc, free

cdef int *a = <int *>malloc(n * sizeof(int))
cdef int *b = <int *>malloc(n * sizeof(int))
if not a or not b:
    raise MemoryError()  # leaks a if b failed
```

**GOOD** -- cascading cleanup:
```cython
from libc.stdlib cimport malloc, free

cdef int *a = NULL
cdef int *b = NULL
try:
    a = <int *>malloc(n * sizeof(int))
    if not a:
        raise MemoryError()
    b = <int *>malloc(n * sizeof(int))
    if not b:
        raise MemoryError()
    # ... use a and b ...
finally:
    if b: free(b)
    if a: free(a)
```

### Stack vs heap: when to use fixed-size C arrays

For small, fixed-size buffers, use stack-allocated C arrays instead of malloc.
No allocation overhead, no cleanup needed, and no risk of leaks.

**GOOD** -- stack allocation for small known sizes:
```cython
# Stack-allocated: fast, no cleanup needed
cdef int counts[256]   # fine for small arrays
cdef double buf[64]

# Use memset for initialization
from libc.string cimport memset
memset(counts, 0, 256 * sizeof(int))
```

**GOOD** -- heap allocation for large or dynamic sizes:
```cython
from libc.stdlib cimport malloc, free

# Heap: required for runtime-sized or large allocations
cdef int *data = <int *>malloc(n * sizeof(int))
```

Rule of thumb: stack for sizes known at compile time and under ~4KB; heap for
everything else.

### memset and memcpy usage

Always import from `libc.string`, not `libc.stdlib` (see [[c-interop]] for
the common `'libc/stdlib/memset.pxd' not found` error).

```cython
from libc.string cimport memset, memcpy

# Zero-initialize an array
cdef int arr[256]
memset(arr, 0, 256 * sizeof(int))

# Copy between buffers
cdef int src[10]
cdef int dst[10]
memcpy(dst, src, 10 * sizeof(int))
```

## Gotchas

1. **Missing cimport is the #1 error.** `malloc` and `free` are NOT builtins.
   You must write `from libc.stdlib cimport malloc, free`. The error message
   `undeclared name not builtin: malloc` appears across all problem categories.

2. **Wrong import path for memset/memcpy.** `memset` lives in `libc.string`,
   not `libc.stdlib`. Writing `from libc.stdlib cimport memset` gives
   `'libc/stdlib/memset.pxd' not found`. See [[c-interop]].

3. **No NULL check = segfault.** Graph problems had 137 segfaults in 500
   traces. Many are caused by dereferencing a NULL pointer from a failed
   malloc. Always check.

4. **Early return without free.** If any code path between `malloc` and `free`
   can raise an exception or return early, you have a memory leak. Use
   `try/finally`.

5. **realloc pitfall.** Never write `arr = realloc(arr, ...)`. If realloc
   fails and returns NULL, you lose the original pointer. Always assign to a
   temporary first.

6. **Non-extern C function declared but not defined.** If you declare a `cdef`
   function (e.g., `cdef void _free_tree(...)`) but forget to implement it,
   you get a linker-style error. Every `cdef` function needs a body.

7. **Recursive free in tree/graph structures.** In graph problems, manually
   freeing tree nodes requires visiting every node. Prefer iterative free with
   a stack to avoid C stack overflow on deep trees.

## See Also

- [[c-interop]] -- cimport paths, struct definitions, extern declarations
- [[extension-types]] -- `__cinit__`/`__dealloc__` for RAII cleanup
- [[memoryviews]] -- typed memoryviews as a safer alternative to raw pointers
- [[pitfalls]] -- general compilation and runtime error patterns
- [[error-handling]] -- exception safety with C memory
