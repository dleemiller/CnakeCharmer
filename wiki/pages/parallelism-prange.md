# Parallelism: prange

Core `prange` usage patterns.

## Pattern

```cython
from cython.parallel cimport prange

cdef Py_ssize_t i
cdef double total = 0.0
for i in prange(n, nogil=True):
    total += a[i] * b[i]
```

## Gotchas

- `reduction=` keyword is invalid in Cython `prange`.
- Python list/dict/tuple operations are disallowed in `nogil` loops.
- Ensure index and accumulator variables are C-typed.

## See Also

- [[parallelism-nogil-callbacks]]
- [[compiler-directives-performance]]
