# Memory Management: Cleanup Strategy

Reliable cleanup for single and multi-buffer C allocations.

## Overview

The safest default is `try/finally` around every owned allocation.

## Single Buffer Pattern

```cython
from libc.stdlib cimport malloc, free

cdef double *buf = <double *>malloc(n * sizeof(double))
if not buf:
    raise MemoryError()

try:
    # use buf
    pass
finally:
    free(buf)
```

## Multi Buffer Pattern

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
finally:
    if b:
        free(b)
    if a:
        free(a)
```

## Gotchas

- Early returns must still pass through cleanup.
- Do not double-free pointers.
- Initialize pointers to NULL before allocation.

## See Also

- [[memory-management-allocation]]
- [[error-handling-c-cleanup]]
- [[pitfalls]]
