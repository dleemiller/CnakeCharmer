# Memory Management: Allocation Patterns

Allocation patterns for `malloc`, `calloc`, and `realloc`.

## Overview

Allocation correctness is non-negotiable in Cython. Missing checks are a common source of segfaults.

## Syntax

```cython
from libc.stdlib cimport malloc, free, calloc, realloc

cdef int *a = <int *>malloc(n * sizeof(int))
if not a:
    raise MemoryError()

cdef int *b = <int *>calloc(n, sizeof(int))
if not b:
    free(a)
    raise MemoryError()
```

## realloc Pattern

```cython
cdef int *tmp = <int *>realloc(a, new_n * sizeof(int))
if not tmp:
    free(a)
    raise MemoryError()
a = tmp
```

## Gotchas

- Never use pointer data before NULL check.
- `realloc` can fail and return NULL while original pointer remains valid.
- Keep allocation and ownership logic obvious in one scope.

## See Also

- [[memory-management-cleanup]]
- [[c-interop-imports-paths]]
- [[pitfalls]]
