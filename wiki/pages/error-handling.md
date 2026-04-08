# Error Handling

Exception return values, propagation through nogil blocks, and C-level error handling.

## Overview

`cdef` functions need explicit error return specifications so Cython knows
how to propagate exceptions across the C call boundary. Without them,
exceptions inside `cdef` functions are **silently swallowed** -- one of the
most dangerous Cython pitfalls. Choosing the right except clause also has
direct performance implications: some forms require acquiring the GIL on
every call, which defeats the purpose of nogil optimization.

## Exception Clause Syntax

### `except <value>` -- Sentinel Value

The function returns a fixed sentinel on error. Cython checks the return
value and only calls `PyErr_Occurred()` when it matches the sentinel.
**Cheapest option** -- zero overhead when no exception occurs and the return
value differs from the sentinel.

```cython
cdef int binary_search(int *arr, int n, int target) except -1:
    if n <= 0:
        raise ValueError("empty array")
    # ... search logic ...
    return index
```

Use this when you can guarantee a value the function never returns normally
(e.g., `-1` for non-negative indices, `NAN` for doubles).

### `except? <value>` -- Checked Sentinel

Like `except <value>`, but Cython always calls `PyErr_Occurred()` when the
return matches the sentinel, even if no exception was set. Small overhead
from the extra check, but necessary when the sentinel is a valid return
value.

```cython
from libc.math cimport sqrt, NAN

cdef double safe_sqrt(double x) except? -1.0:
    if x < 0:
        raise ValueError("negative input")
    return sqrt(x)
```

### `except *` -- Always Check

Cython calls `PyErr_Occurred()` after **every** call. Safest but slowest
option. Required when any return value is valid and no sentinel exists.

```cython
cdef bint contains(int *arr, int n, int val) except *:
    cdef int i
    for i in range(n):
        if arr[i] == val:
            return True
    return False
```

**Performance note**: `except *` always requires the GIL to call
`PyErr_Occurred()`. If used on a function called inside a `prange` or
`nogil` block, Cython must re-acquire the GIL on every call. Traces show
this as:

> `performance hint: Exception check on 'compute_rhs' will always require the GIL to be acquired.`

### `noexcept` -- No Exception Checking

Tells Cython the function will never raise a Python exception. Zero
overhead, fully compatible with `nogil`, but **any exception raised inside
the function is silently lost**.

```cython
cdef inline double fast_dot(double *a, double *b, int n) noexcept nogil:
    cdef double s = 0.0
    cdef int i
    for i in range(n):
        s += a[i] * b[i]
    return s
```

### C++ Exception Translation

```cython
cdef extern from "lib.h":
    double divide(double a, double b) except +               # any C++ exception -> RuntimeError
    int lookup(int key) except +IndexError                    # map to specific Python exception
    void process() except +*                                  # custom handler
```

## Decision Tree: Which Except Form to Use

```
Is the function called from nogil / prange context?
|
+-- YES --> Can the function actually raise Python exceptions?
|           |
|           +-- NO  --> Use `noexcept`
|           +-- YES --> Can you acquire the GIL inside to handle it?
|                       |
|                       +-- YES --> Use `except <sentinel>` + `with gil:` for the raise
|                       +-- NO  --> Use `noexcept`, handle errors via C return codes
|
+-- NO --> Does the function return a C type?
           |
           +-- YES --> Is there a value the function never returns normally?
           |           |
           |           +-- YES --> Use `except <sentinel>` (fastest)
           |           +-- NO  --> Is there a value that is *rarely* returned?
           |                       |
           |                       +-- YES --> Use `except? <value>`
           |                       +-- NO  --> Use `except *` (accept GIL cost)
           |
           +-- NO (returns Python object) --> Do NOT add an except clause.
                                              Python objects propagate exceptions
                                              automatically via NULL return.
```

## Common Error: "Exception clause not allowed for function returning Python object"

This is one of the most frequent error-handling compilation errors in the
traces. It happens when you add an except clause to a function that returns
a Python object (`list`, `dict`, `object`, `str`, etc.):

```cython
# BAD -- Python objects already use NULL as the error sentinel
cdef list get_neighbors(int node) except *:
    ...

# BAD -- same problem with noexcept on Python-returning function
cdef object parse_data(str text) noexcept:
    ...
```

```cython
# GOOD -- no except clause needed for Python object returns
cdef list get_neighbors(int node):
    ...

cdef object parse_data(str text):
    ...
```

Cython automatically returns `NULL` for Python objects on exception, and
the caller checks for `NULL`. Adding any except clause (`except *`,
`except -1`, or `noexcept`) is a compilation error.

**Trace evidence**: This error appeared across diff_equations (5 instances),
physics (3 instances), and algorithms categories.

## Exception Propagation Through nogil Blocks

When a `cdef noexcept nogil` function encounters a situation that should be
an error, you cannot raise a Python exception directly. Use the
`with gil:` pattern to temporarily re-acquire the GIL:

```cython
# BAD -- raises inside nogil without acquiring GIL
cdef double compute_step(double *y, int n) noexcept nogil:
    if n <= 0:
        raise ValueError("bad size")  # Compile error: cannot raise without GIL
    ...
```

```cython
# GOOD -- acquire GIL to raise, use except clause to propagate
cdef double compute_step(double *y, int n) except? -1.0 nogil:
    if n <= 0:
        with gil:
            raise ValueError("bad size")
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += y[i] * y[i]
    return result
```

Note the function uses `except? -1.0` (not `noexcept`), because it *can*
raise via the `with gil:` block. Using `noexcept` here would silently
swallow the exception.

**Pattern from traces**: The `with gil:` re-acquire pattern appears
frequently in diff_equations (method_of_lines, midpoint_method) and physics
(blackbody_radiation, capacitor_discharge) problems. Fix success rate for
GIL violations was 67-86% across categories.

## C Callback Functions Need `noexcept`

When passing a Cython function as a C callback (e.g., to `qsort`, a
numerical integrator, or any C library expecting a function pointer), the
function **must** be declared `noexcept`. C function pointers have no
concept of Python exceptions.

```cython
from libc.stdlib cimport qsort

# BAD -- default except clause makes this incompatible with C function pointers
cdef int compare(const void *a, const void *b):
    return (<int*>a)[0] - (<int*>b)[0]

# Compiler error:
# Cannot assign type 'int (*)(const void *, const void *) except *' to
# 'int (*)(const void *, const void *)'
qsort(arr, n, sizeof(int), compare)
```

```cython
# GOOD -- noexcept makes the signature match the C function pointer
cdef int compare(const void *a, const void *b) noexcept:
    return (<int*>a)[0] - (<int*>b)[0]

qsort(arr, n, sizeof(int), compare)
```

**Trace evidence**: This exact error ("Cannot assign type 'int (...) except *'
to 'int (...)'") appeared in the algorithms category (e.g., sorting
problems using `qsort`). The fix is always to add `noexcept`.

## C Library Error Handling: malloc NULL Checks and errno

When calling C library functions, you must check for errors using C
conventions -- return codes, NULL pointers, and `errno` -- rather than
relying on Python exception mechanisms.

### malloc NULL Checks

```cython
from libc.stdlib cimport malloc, free

cdef double* allocate_buffer(int n) except NULL:
    cdef double *buf = <double*>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError("Failed to allocate buffer")
    return buf
```

Note: `except NULL` works for pointer-returning functions, using `NULL` as
the sentinel value. See [[memory-management]] for full allocation patterns.

### errno Checking

```cython
from libc.errno cimport errno
from libc.string cimport strerror
from libc.stdio cimport fopen, fclose, FILE

cdef FILE* safe_open(const char *path) except NULL:
    cdef FILE *f = fopen(path, "rb")
    if f == NULL:
        raise OSError(errno, strerror(errno).decode('utf-8'))
    return f
```

## try/finally for Cleanup with C Allocations

When mixing C allocations with code that might raise exceptions, use
`try/finally` to guarantee cleanup. This is critical because Python's
garbage collector does not manage C memory.

```cython
# BAD -- leak on exception
cdef list process_data(double[:] input_view):
    cdef int n = input_view.shape[0]
    cdef double *temp = <double*>malloc(n * sizeof(double))
    if temp == NULL:
        raise MemoryError()

    # If this raises, temp is leaked
    result = do_computation(temp, n)
    free(temp)
    return result
```

```cython
# GOOD -- try/finally guarantees cleanup
cdef list process_data(double[:] input_view):
    cdef int n = input_view.shape[0]
    cdef double *temp = <double*>malloc(n * sizeof(double))
    if temp == NULL:
        raise MemoryError()

    try:
        result = do_computation(temp, n)
        return result
    finally:
        free(temp)
```

For functions managing multiple allocations, nest the try/finally blocks or
use a single cleanup block:

```cython
cdef object solve_system(int n):
    cdef double *a = <double*>malloc(n * sizeof(double))
    cdef double *b = NULL
    if a == NULL:
        raise MemoryError()
    try:
        b = <double*>malloc(n * sizeof(double))
        if b == NULL:
            raise MemoryError()
        # ... computation ...
        return build_result(a, b, n)
    finally:
        free(a)
        if b != NULL:
            free(b)
```

See [[memory-management]] for more allocation patterns including RAII-style
wrappers.

## noexcept vs except * Tradeoffs

| Aspect | `noexcept` | `except *` |
|---|---|---|
| GIL required on call | No | Yes (calls `PyErr_Occurred()`) |
| nogil compatible | Yes | No (forces GIL re-acquire) |
| prange compatible | Yes | No |
| C callback compatible | Yes | No |
| Exception safety | Exceptions silently lost | All exceptions propagated |
| Best for | Hot inner loops, callbacks | Functions that might raise |

**When to prefer `noexcept`**:
- Pure C arithmetic in tight loops (inner kernels)
- Functions called from `prange` parallel loops
- C callback functions passed to `qsort`, integrators, etc.
- Functions where you have verified no Python exception can occur

**When to prefer `except *` (or a sentinel form)**:
- Functions that validate input and may raise
- Functions that call Python/NumPy operations
- Functions where silently lost exceptions would cause subtle bugs
- Wrapper functions at the boundary between C and Python code

**The middle ground**: Use `except <sentinel>` when possible. It gives you
exception safety with minimal overhead (one comparison per call, no GIL
acquire). Common sentinel choices:

- `except -1` for int functions returning non-negative values
- `except? -1.0` for double functions (any value could be valid)
- `except NULL` for pointer-returning functions

## Trace Statistics

Analyzed **1,039 traces** across 3 categories (diff_equations, physics,
algorithms):

| Category | Traces | Error-Handling Errors | GIL Violations | Fix Rate |
|---|---|---|---|---|
| diff_equations | 314 | 6 (except clause on Python obj, noexcept syntax) | 6 | 67% |
| physics | 225 | 4 (except clause on Python obj, noexcept syntax) | 8 | 86% |
| algorithms | 500 | 2 (callback type mismatch) | 44 | 84% |
| **Total** | **1,039** | **12** | **58** | **80% avg** |

**Most common error-handling mistakes** (by frequency):
1. **"Exception clause not allowed for function returning Python object"** -- 9 instances across diff_equations + physics. Models add `except *` or `except -1` to `cdef` functions returning `list`, `object`, etc.
2. **"Exception check will always require the GIL"** -- 5 performance hints in diff_equations + physics. Models use `except *` on functions intended for nogil hot loops.
3. **"Cannot assign type ... except * to ..."** -- 2 instances in algorithms. Models forget `noexcept` on C callback functions.
4. **"Expected ':', found 'noexcept'"** -- 2 instances. Models place `noexcept` in wrong position (after return type instead of after parameters).

**GIL violation resolution**: Across all categories, 58 traces hit GIL
violations related to exception handling. The fix success rate averaged 80%,
with the most common fix being addition of `noexcept` or switching from
`except *` to a sentinel form.

## See Also

- [[parallelism]] -- GIL interaction with exception clauses in prange loops
- [[memory-management]] -- malloc/free patterns and try/finally cleanup
- [[pitfalls]] -- Common compilation errors including except clause mistakes
- [[typing]] -- cdef function declarations and return types
- [[cpp-interop]] -- C++ exception translation (`except +`)
