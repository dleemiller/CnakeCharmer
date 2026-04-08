# Error Handling: C Allocation Cleanup

Avoid leaks and partial-failure bugs when using `malloc`/`free` in Cython.

## Overview

Manual allocation errors are a top source of crashes and leaks. Always pair allocation with deterministic cleanup.

## Core Pattern

```cython
from libc.stdlib cimport malloc, free

cdef int *buf = <int *>malloc(n * sizeof(int))
if not buf:
    raise MemoryError()

try:
    # use buf
    pass
finally:
    free(buf)
```

## Multiple Allocation Pattern

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

    # use a, b
finally:
    if b:
        free(b)
    if a:
        free(a)
```

## Gotchas

- Never skip NULL checks.
- Never return early before cleanup.
- For `realloc`, keep old pointer until success is confirmed.

## See Also

- [[memory-management-allocation]]
- [[memory-management-cleanup]]
- [[pitfalls]]
