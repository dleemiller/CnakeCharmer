# Error Handling: nogil and C Callbacks

Error propagation rules for `nogil` code and callback function pointers.

## Overview

Two high-frequency issues from traces:
- Raising from `nogil` code without reacquiring the GIL.
- Passing non-`noexcept` Cython functions to C callback APIs.

## Patterns

### Raise from `nogil` safely

```cython
cdef double step(double *x, int n) except? -1.0 nogil:
    if n <= 0:
        with gil:
            raise ValueError("n must be positive")
    return x[0]
```

### Callback signatures must match C type

```cython
from libc.stdlib cimport qsort

cdef int cmp_int(const void *a, const void *b) noexcept nogil:
    cdef int ai = (<int *>a)[0]
    cdef int bi = (<int *>b)[0]
    return (ai > bi) - (ai < bi)

# Compatible callback
qsort(arr, n, sizeof(int), cmp_int)
```

## Gotchas

- Non-`noexcept` callback functions often compile as `except *`, causing type mismatch with C function pointers.
- `except *` in tight `nogil` loops can degrade performance due to mandatory checks.
- If callbacks cannot raise, keep them `noexcept nogil` and return error codes explicitly.

## See Also

- [[c-interop-function-pointers]]
- [[parallelism]]
- [[compiler-directives]]
