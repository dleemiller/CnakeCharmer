# Typing

Static type declarations for C-speed execution.

## Overview

Cython's main performance lever: declaring C types removes Python object overhead
from hot paths. Variables, function parameters, and return types can all be typed.

## Syntax

```cython
# Variable declarations (must be at function top, before executable code)
cdef int i
cdef double total = 0.0
cdef long long big_num

# Function declarations
cdef double fast_internal(double x, double y):    # C-only, not callable from Python
    return x * y

cpdef double both_ways(double x, double y):       # callable from C and Python
    return x * y

def python_visible(int n):                        # Python-callable with typed params
    cdef int i
    for i in range(n): ...

# Fused types (generics)
ctypedef fused number_t:
    int
    long long
    float
    double

def generic_sum(number_t[:] arr):
    cdef number_t total = 0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# Type aliases
ctypedef unsigned long long uint64
ctypedef double (*func_ptr)(double)
```

## Patterns

### Declare All C Variables at Function Top

All `cdef` declarations must appear before any executable statements in a function body. This is the single most common typing error in traces (229+ occurrences across categories).

```cython
# BAD — cdef after executable code
def func(int n):
    result = []
    cdef int i          # ERROR: cdef statement not allowed here

# GOOD — all cdef declarations at the top
def func(int n):
    cdef int i
    result = []
```

### Variable Redeclaration

Declaring the same variable twice in different scopes causes errors.

```cython
# BAD — redeclared variable
def func(int n):
    cdef int idx
    for idx in range(n):
        ...
    cdef int idx        # ERROR: 'idx' redeclared

# GOOD — declare once, reuse
def func(int n):
    cdef int idx
    for idx in range(n):
        ...
    # reuse idx freely after loop
```

### Special Methods Must Use `def`

Cython extension type special methods (`__len__`, `__getitem__`, etc.) must be declared with `def`, not `cdef` or `cpdef`.

```cython
# BAD — special methods cannot be cdef
cdef class MyContainer:
    cdef int __len__(self):       # ERROR: Special methods must be declared with 'def', not 'cdef'
        return self.size

# GOOD
cdef class MyContainer:
    def __len__(self):
        return self.size
```

### Closures Inside cpdef

`cpdef` functions cannot contain closures or inner functions that capture local variables.

```cython
# BAD — closure inside cpdef
cpdef list sort_custom(list items):
    def key_func(x):       # ERROR: closures not supported in cpdef
        return x.value
    return sorted(items, key=key_func)

# GOOD — use def instead, or extract to module-level
def sort_custom(list items):
    def key_func(x):
        return x.value
    return sorted(items, key=key_func)
```

### Syntax Errors in C Variable Declarations

Using Python-style type hints or invalid C syntax in `cdef` declarations. This is the top compilation error (9 occurrences in algorithms alone).

```cython
# BAD — Python annotation syntax in cdef
cdef int x: int = 0              # ERROR: Syntax error in C variable declaration

# BAD — missing type
cdef = 0                         # ERROR

# GOOD — standard cdef syntax
cdef int x = 0
```

### Storing Unsafe C Derivative of Temporary Python Reference

Taking a C pointer to a temporary Python object that may be garbage collected.

```cython
# BAD — pointer to temporary
cdef const char* s = some_string.encode('utf-8')  # WARNING: Storing unsafe C derivative of temporary Python reference

# GOOD — keep a reference alive
py_bytes = some_string.encode('utf-8')
cdef const char* s = py_bytes
```

### Type Identifiers Require cimport

Using C++ STL types or external types requires proper cimport.

```cython
# BAD — using type without cimport
cdef priority_queue[int] pq     # ERROR: 'priority_queue' is not a type identifier

# GOOD — cimport first
from libcpp.queue cimport priority_queue
cdef priority_queue[int] pq
```

### Multiplied List to Memoryview Coercion

Python list expressions cannot be directly assigned to typed memoryviews.

```cython
# BAD — list literal to memoryview
cdef int[:] arr = [0] * n       # ERROR: cannot coerce list to memoryview

# GOOD — create numpy array, then assign
import numpy as np
cdef int[:] arr = np.zeros(n, dtype=np.intc)
```

### Python Object to C Pointer Conversion

Cannot implicitly convert Python objects to C pointers.

```cython
# BAD — implicit conversion
cdef int* ptr = python_list     # ERROR: Cannot convert Python object to pointer

# GOOD — use typed memoryview or malloc
cdef int[:] view = np.array(python_list, dtype=np.intc)
# or
cdef int* ptr = <int*>malloc(n * sizeof(int))
```

## Trace Statistics

Across 500 algorithm traces (representative sample):

| Error Type | Count | Fix Rate |
|-----------|-------|----------|
| cdef placement | 77 | ~72% (single-step) |
| Syntax in C variable decl | 9+ | High |
| Unsafe temporary reference | 3+ | High |
| Type identifier missing | 4+ | High |
| Variable redeclaration | 2+ | High |
| Special method with cdef | 2+ | High |

**Key insight**: cdef placement is the #1 typing error. Always declare all C variables at the function top, before any assignments, calls, or control flow.

## Gotchas

1. **cdef before code** — Every `cdef` must precede all executable statements. Even `x = 0` counts as executable.
2. **No redeclaration** — Cannot `cdef` the same variable name twice in one scope.
3. **int overflow** — Python `int` is arbitrary precision; C `int` overflows silently. Use `long long` for large values.
4. **fused types need memoryviews** — `ctypedef fused` works with typed memoryviews, not raw Python lists.
5. **`def` for special methods** — `__init__`, `__len__`, `__getitem__` etc. must always use `def`.
6. **cpdef limitations** — No closures, no `*args`/`**kwargs`, no generators.
7. **cimport vs import** — C types need `cimport`; regular Python imports won't make C types available.
8. **Temporary references** — Keep Python objects alive when taking C pointers to their data.
9. **Multiplied lists** — `[0] * n` is a Python list, not a C array. Use numpy or malloc.

## See Also

[[extension-types]], [[compiler-directives]], [[enums-tuples]], [[pitfalls]], [[c-interop]]
