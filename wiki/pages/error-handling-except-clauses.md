# Error Handling: Except Clauses

How to choose `except`, `except?`, `except *`, and `noexcept` for C-returning Cython functions.

## Overview

For `cdef` functions returning C types, Cython needs to know how exceptions cross the C boundary.
The wrong choice can either hide errors or add avoidable overhead.

## Syntax

```cython
cdef int f1(int x) except -1:
    ...

cdef double f2(double x) except? -1.0:
    ...

cdef bint f3(int *arr, int n) except *:
    ...

cdef int f4(int x) noexcept:
    ...
```

## Decision Rules

1. Prefer `except <sentinel>` when you have a value that cannot appear in valid output.
2. Use `except? <sentinel>` when sentinel can appear in valid output and ambiguity must be resolved.
3. Use `except *` only when no sentinel exists and all values are valid.
4. Use `noexcept` only if function cannot raise Python exceptions.

## Patterns

### Sentinel fast path

```cython
cdef int find_idx(int *arr, int n, int target) except -1:
    cdef int i
    for i in range(n):
        if arr[i] == target:
            return i
    raise ValueError("target not found")
```

### Checked sentinel

```cython
from libc.math cimport sqrt

cdef double safe_sqrt(double x) except? -1.0:
    if x < 0:
        raise ValueError("negative")
    return sqrt(x)
```

## Gotchas

- Do not add `except` to functions returning Python objects (`list`, `dict`, `object`, `str`).
- `except *` may require extra error checks and can force GIL interaction in hot call paths.
- `noexcept` suppresses exception propagation; raising inside it is a logic error.

## See Also

- [[error-handling-nogil-callbacks]]
- [[parallelism]]
- [[pitfalls]]
