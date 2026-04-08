# Enums and C-Tuples

Lightweight C-level enumerations and tuple return types.

## Overview

`cdef enum` creates C-level constants (compile-time, zero overhead).
`cpdef enum` adds Python visibility. C-tuples return multiple values
without Python tuple allocation overhead.

## Syntax

```cython
# cdef enum — C-only constants
cdef enum Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# Anonymous enum — just constants
cdef enum:
    MAX_SIZE = 1024
    HASH_MASK = 0xFF

# cpdef enum — accessible from Python too
cpdef enum TokenType:
    INTEGER = 0
    OPERATOR = 1
    PAREN = 2

# C-tuples — return multiple values without Python overhead
cdef (double, double) minmax(double *arr, int n):
    cdef double mn = arr[0], mx = arr[0]
    for i in range(1, n):
        if arr[i] < mn: mn = arr[i]
        if arr[i] > mx: mx = arr[i]
    return (mn, mx)

cdef (long long, long long) divmod_c(long long a, long long b):
    return (a // b, a % b)
```

## Patterns

### Enum as Compile-Time Constants

Enums replace magic numbers with named constants at zero cost:

```cython
cdef enum:
    BUCKET_COUNT = 256
    SEED = 0x9E3779B1

def hash_histogram(unsigned char[::1] data):
    cdef int counts[BUCKET_COUNT]    # uses enum constant
    cdef int i, n = data.shape[0]
    memset(counts, 0, BUCKET_COUNT * sizeof(int))
    for i in range(n):
        counts[data[i]] += 1
    return [counts[i] for i in range(BUCKET_COUNT)]
```

### cpdef enum for Python-Visible Constants

When Python code needs to reference the enum values:

```cython
cpdef enum Status:
    OK = 0
    ERROR = 1
    TIMEOUT = 2

def check_result(int code):
    if code == Status.OK:       # works from both Cython and Python
        return True
    return False

# From Python:
# from module import Status
# Status.OK  → 0
```

### C-Tuples for Multi-Return Without GIL

Python tuples are Python objects and require the GIL. C-tuples are pure C:

```cython
# BAD — Python tuple in nogil block
cdef object find_minmax(double *arr, int n) nogil:
    # ERROR: cannot return Python object without GIL
    return (arr[0], arr[1])

# GOOD — C-tuple works in nogil
cdef (double, double) find_minmax(double *arr, int n) noexcept nogil:
    cdef double mn = arr[0], mx = arr[0]
    cdef int i
    for i in range(1, n):
        if arr[i] < mn: mn = arr[i]
        if arr[i] > mx: mx = arr[i]
    return (mn, mx)

# Usage from another nogil function:
cdef void process(double *data, int n) noexcept nogil:
    cdef (double, double) result = find_minmax(data, n)
    cdef double lo = result[0]
    cdef double hi = result[1]
```

This pattern avoids the "Constructing Python tuple not allowed without gil" error that appeared 20+ times in numerical traces. See [[parallelism]].

### Enum Flags and Bit Manipulation

```cython
cdef enum Flags:
    FLAG_READ = 1
    FLAG_WRITE = 2
    FLAG_EXEC = 4

cdef bint has_flag(int flags, int flag) noexcept nogil:
    return (flags & flag) != 0

cdef int set_flag(int flags, int flag) noexcept nogil:
    return flags | flag
```

### Struct vs C-Tuple for Multiple Values

For more than 2-3 values, or when values have names, prefer a struct:

```cython
# C-tuple — fine for 2 values
cdef (int, double) find_best(double *arr, int n) noexcept nogil:
    ...

# Struct — better for 3+ values or named fields
cdef struct SearchResult:
    int index
    double value
    bint found

cdef SearchResult find_best(double *arr, int n, double target) noexcept nogil:
    cdef SearchResult r
    r.found = False
    # ...
    return r
```

## Gotchas

1. **cdef enum is C-only** — `cdef enum` values are not accessible from Python. Use `cpdef enum` for Python visibility.
2. **No string enums** — Cython enums are integer-only (like C enums). For string constants, use module-level `cdef` variables.
3. **C-tuples need parentheses in type** — Declare as `cdef (int, int) result`, not `cdef int, int result`.
4. **C-tuples are not Python tuples** — They can't be passed to Python functions or stored in Python containers without conversion.
5. **Tuple construction in nogil** — Use C-tuples, not Python tuples, inside `nogil` or `prange` blocks.

## See Also

[[typing]], [[c-interop]], [[parallelism]]
