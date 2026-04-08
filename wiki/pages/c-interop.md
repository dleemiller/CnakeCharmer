# C Interop

Calling C libraries, using structs, unions, and function pointers.

## Overview

Cython can directly call C functions with zero overhead. `cimport` pulls in C
declarations, `cdef extern` wraps custom headers, and C structs/unions map to
Cython types. This is how you access libc math, stdlib, and custom C code.

## Trace Statistics

| Source | Traces | Compile Errors | cimport Path Errors | `'X' is not a cimported module` |
|--------|--------|---------------|--------------------|---------------------------------|
| algorithms | 500 | 214 | 11 | 8 |
| graph | 500 | 457 | 6 | 21 |
| cryptography | 408 | 261 | -- | -- |
| **Total** | **1408** | **932** | **~17** | **~29** |

Top c-interop compile errors across 1408 traces:
- `'np' is not a cimported module` -- 14+ instances (use `cimport numpy as np`)
- `'libc/stdlib/memset.pxd' not found` -- 10 instances (memset is in `libc.string`)
- `'array' is not a cimported module` -- 12+ instances
- `'libc/stddef/NULL.pxd' not found` -- 4 instances (NULL needs no import)
- `'cpython/object/PyLong_FromLong.pxd' not found` -- 3 instances (wrong granularity)
- `'malloc' not a valid cython language construct` -- 1 instance (used without cimport)
- Function pointer `except *` vs `noexcept` type mismatches -- 9 instances

## Syntax

```cython
# libc imports -- the correct paths
from libc.math cimport sin, cos, sqrt, log, exp, M_PI, INFINITY
from libc.stdlib cimport malloc, free, calloc, realloc, qsort, abs as c_abs
from libc.string cimport memcpy, memset, memcmp, strlen
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t, int64_t
from cpython.object cimport PyObject

# Structs
cdef struct Point:
    double x
    double y

cdef struct Rect:
    Point top_left
    Point bottom_right    # nested struct

# Struct -> dict conversion (automatic in Cython)
def make_point(double x, double y) -> dict:
    cdef Point p = Point(x=x, y=y)
    return p  # auto-converts to dict

# Unions
cdef union IntFloat:
    int as_int
    float as_float

# Function pointers
ctypedef int (*comparator)(const void *, const void *) noexcept nogil

cdef int cmp_double(const void *a, const void *b) noexcept nogil:
    cdef double da = (<double *>a)[0]
    cdef double db = (<double *>b)[0]
    return (da > db) - (da < db)

# Using qsort
qsort(<void *>arr, n, sizeof(double), cmp_double)

# Extern from header
cdef extern from "mylib.h":
    double my_function(double x)

# Extern for inline C
cdef extern from *:
    """
    static inline int fast_popcount(unsigned int x) {
        return __builtin_popcount(x);
    }
    """
    int fast_popcount(unsigned int x) noexcept nogil
```

## Patterns

### cimport vs import: the critical distinction

`cimport` brings in C-level declarations for use in `cdef` typed code.
`import` brings in Python-level modules. Confusing them is a top error source.

**BAD** -- using `import` where `cimport` is needed:
```cython
import numpy as np

def typed_sum(double[:] arr):
    cdef np.npy_intp i  # ERROR: 'np' is not a cimported module
    cdef double total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total
```

**GOOD** -- both import and cimport for NumPy:
```cython
import numpy as np
cimport numpy as np

def typed_sum(double[:] arr):
    cdef np.npy_intp i
    cdef double total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total
```

### Correct cimport paths for libc

Cython ships `.pxd` files organized by C header. The import path mirrors the
header, NOT the function name.

**BAD** -- wrong granularity, treating functions as submodules:
```cython
from libc.stdlib cimport memset      # ERROR: 'libc/stdlib/memset.pxd' not found
from libc.stddef cimport NULL        # ERROR: 'libc/stddef/NULL.pxd' not found
from cpython.object cimport PyLong_FromLong  # ERROR: wrong path
```

**GOOD** -- correct paths:
```cython
from libc.string cimport memset      # memset is in <string.h>, not <stdlib.h>
from libc.stdlib cimport malloc, free # malloc/free are in <stdlib.h>
from libc.math cimport sqrt, sin     # math functions in <math.h>

# NULL is available automatically after any libc cimport, or:
from libc.stdint cimport uint8_t     # stdint types
```

Reference table of common cimport paths:

| Function/Type | Correct cimport | C Header |
|---------------|----------------|----------|
| `malloc`, `free`, `calloc`, `realloc` | `from libc.stdlib cimport ...` | `<stdlib.h>` |
| `qsort`, `abs` | `from libc.stdlib cimport ...` | `<stdlib.h>` |
| `memcpy`, `memset`, `memcmp`, `strlen` | `from libc.string cimport ...` | `<string.h>` |
| `sin`, `cos`, `sqrt`, `log`, `exp` | `from libc.math cimport ...` | `<math.h>` |
| `M_PI`, `INFINITY`, `NAN` | `from libc.math cimport ...` | `<math.h>` |
| `uint8_t`, `uint32_t`, `int64_t` | `from libc.stdint cimport ...` | `<stdint.h>` |
| `FILE`, `fopen`, `fclose` | `from libc.stdio cimport ...` | `<stdio.h>` |
| `PyObject` | `from cpython.object cimport ...` | `<Python.h>` |
| `Py_INCREF`, `Py_DECREF` | `from cpython.ref cimport ...` | `<Python.h>` |

### NULL does not need its own cimport

**BAD**:
```cython
from libc.stddef cimport NULL  # ERROR: 'libc/stddef/NULL.pxd' not found
```

**GOOD** -- NULL is a Cython builtin when you have any libc cimport:
```cython
from libc.stdlib cimport malloc, free

cdef int *p = <int *>malloc(10 * sizeof(int))
if p == NULL:
    raise MemoryError()
```

Or just use the Pythonic `not` check:
```cython
if not p:
    raise MemoryError()
```

### Struct definitions and usage

Structs are declared with `cdef struct` at module level. They can be nested,
passed by pointer, and automatically convert to/from Python dicts.

```cython
cdef struct Node:
    int value
    int left
    int right

cdef struct Edge:
    int src
    int dst
    double weight

# Initialize with keyword args
cdef Node n = Node(value=10, left=-1, right=-1)

# Or positional
cdef Edge e = Edge(0, 1, 3.14)

# Pass by pointer
cdef void process_node(Node *n) noexcept nogil:
    n.value += 1
```

### Wrapping C functions with extern

Use `cdef extern from` to wrap functions from C headers or embed inline C.

```cython
# From a header file
cdef extern from "fast_hash.h":
    uint64_t hash64(const void *data, size_t len) noexcept nogil

# Inline C (no header needed)
cdef extern from *:
    """
    #include <immintrin.h>
    static inline int popcount32(unsigned int x) {
        return __builtin_popcount(x);
    }
    """
    int popcount32(unsigned int x) noexcept nogil
```

### Function pointer signatures: noexcept matters

When passing function pointers to C functions like `qsort`, the exception
specification must match exactly. This caused 9+ compile errors in graph traces.

**BAD** -- exception spec mismatch:
```cython
from libc.stdlib cimport qsort

# Default Cython functions have "except *" -- incompatible with C callbacks
cdef int my_cmp(const void *a, const void *b):  # implicit except *
    return (<int *>a)[0] - (<int *>b)[0]

qsort(arr, n, sizeof(int), my_cmp)  # ERROR: type mismatch
```

**GOOD** -- explicit noexcept nogil:
```cython
from libc.stdlib cimport qsort

cdef int my_cmp(const void *a, const void *b) noexcept nogil:
    return (<int *>a)[0] - (<int *>b)[0]

qsort(arr, n, sizeof(int), my_cmp)  # OK
```

### The `'array' is not a cimported module` error

This appears when trying to use `array` module typed access without proper
cimport.

**BAD**:
```cython
import array

def sum_array(array.array a):  # ERROR: 'array' is not a cimported module
    pass
```

**GOOD**:
```cython
from cpython cimport array
import array

def sum_array(array.array a):
    cdef int[:] view = a
    # work with the memoryview
```

## Gotchas

1. **`'np' is not a cimported module`** is the most frequent c-interop error
   (14+ instances). You must `cimport numpy as np` in addition to
   `import numpy as np` to use `np.` in cdef type declarations. See
   [[numpy-interop]].

2. **`'libc/stdlib/memset.pxd' not found`** appears 10 times across traces.
   `memset`, `memcpy`, and `memcmp` live in `libc.string`, not `libc.stdlib`.
   This mirrors C where they are in `<string.h>`.

3. **Function pointer `except *` vs `noexcept`** type mismatch. C callback
   functions (for `qsort`, custom comparators) must be declared `noexcept
   nogil`. The default Cython exception spec `except *` is incompatible with
   C function pointer types.

4. **`'libc/stddef/NULL.pxd' not found`**. Do not try to cimport NULL
   directly. It is automatically available after any `libc` cimport, or you
   can use the Pythonic `if not ptr:` check.

5. **`'cpython/object/PyLong_FromLong.pxd' not found`**. Individual CPython
   API functions cannot be cimported by name from sub-paths. Import from the
   correct module: `from cpython.long cimport PyLong_FromLong`.

6. **`'malloc' not a valid cython language construct`**. This occurs when
   malloc is used in an expression without any cimport. Cython cannot parse it
   as either a Python name or a C function. Fix: add
   `from libc.stdlib cimport malloc`. See [[memory-management]].

7. **Exception clause on Python-returning functions.** `cdef object func()
   except *` is invalid -- functions returning Python objects always propagate
   exceptions. Omit the except clause for such functions.

8. **`Storing unsafe C derivative of temporary Python reference`**. Assigning
   a C pointer from a temporary Python object (e.g., a bytes string) risks
   dangling pointers. Store the Python object in a variable first.

## See Also

- [[memory-management]] -- malloc/free patterns, NULL checks, try/finally cleanup
- [[cpp-interop]] -- C++ specific: STL containers, classes, templates
- [[numpy-interop]] -- NumPy cimport, typed memoryviews with ndarray
- [[typing]] -- cdef type declarations and placement rules
- [[parallelism]] -- nogil requirements for C callbacks and prange
- [[pitfalls]] -- general compilation error patterns
- [[error-handling]] -- exception specs (noexcept, except *, except? -1)
