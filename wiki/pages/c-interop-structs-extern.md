# C Interop: Structs, Unions, and extern

Defining C data types and declaring external C functions.

## Overview

Use module-level `cdef struct` or `cdef union` definitions and `cdef extern from` for C APIs.

## Syntax

```cython
cdef struct Point:
    double x
    double y

cdef union IntFloat:
    int as_int
    float as_float

cdef extern from "mylib.h":
    double my_function(double x) noexcept nogil
```

## Patterns

### Struct initialization

```cython
cdef Point p = Point(x=1.0, y=2.0)
```

### Inline extern block

```cython
cdef extern from *:
    """
    static inline int fast_popcount(unsigned int x) {
        return __builtin_popcount(x);
    }
    """
    int fast_popcount(unsigned int x) noexcept nogil
```

## Gotchas

- Keep struct declarations at module scope.
- Keep extern signatures aligned with real C signatures.
- Add `noexcept nogil` when functions are used in `nogil` sections.

## See Also

- [[c-interop-function-pointers]]
- [[typing]]
- [[cpp-interop]]
