# C Interop: Function Pointers and Callbacks

Matching C callback signatures in Cython.

## Overview

C callbacks are strict about signature compatibility. In traces, mismatched exception specs were a frequent error.

## Pattern

```cython
from libc.stdlib cimport qsort

ctypedef int (*cmp_fn)(const void *, const void *) noexcept nogil

cdef int cmp_int(const void *a, const void *b) noexcept nogil:
    cdef int ai = (<int *>a)[0]
    cdef int bi = (<int *>b)[0]
    return (ai > bi) - (ai < bi)

qsort(arr, n, sizeof(int), cmp_int)
```

## Gotchas

- Default Cython exception behavior can produce `except *` signatures.
- `except *` callbacks often fail to match pure C function pointers.
- Use `noexcept` for callbacks that must not raise.

## See Also

- [[error-handling-nogil-callbacks]]
- [[parallelism]]
- [[c-interop-imports-paths]]
