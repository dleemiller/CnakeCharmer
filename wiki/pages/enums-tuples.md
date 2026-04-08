# Enums and C-Tuples

Using enums and C-tuples for low-overhead constants and multi-return values.

## Overview

Compact page for practical usage patterns.

## Fast Rules

1. Use `cdef enum` for C-only constants.
2. Use `cpdef enum` when constants must be visible from Python.
3. Use C-tuples in `nogil`/hot C paths for multi-value returns.

## Patterns

```cython
cdef enum Mode:
    FAST = 0
    SAFE = 1

cpdef enum Status:
    OK = 0
    FAIL = 1

cdef (double, double) minmax(double a, double b) noexcept nogil:
    if a < b:
        return a, b
    return b, a
```

## See Also

- [[typing-declarations]]
- [[parallelism-prange]]
- [[optimization-hot-loops]]
