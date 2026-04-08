# Compiler Directives

Control Cython's code generation behavior.

## Overview

Directives disable safety checks for speed or enable special compilation modes.
Set them file-wide via comments, per-function via decorators, or per-block
via `with` statements.

## Syntax

```cython
# File-wide (most common in our codebase)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# Per-function
@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sum(double[::1] arr):
    ...

# Per-block
with cython.boundscheck(False):
    result = arr[i]
```

## Key Directives

| Directive | Default | Effect |
|-----------|---------|--------|
| `boundscheck` | True | Array bounds checking — disable in hot loops |
| `wraparound` | True | Negative index support — disable when not needed |
| `cdivision` | False | C-style division (no ZeroDivisionError, truncates toward zero) |
| `language_level` | 2 | Always set to 3 for Python 3 semantics |
| `initializedcheck` | True | Check memoryviews are initialized before access |
| `nonecheck` | False | Check typed extension types for None before access |
| `overflowcheck` | False | Check C integer arithmetic for overflow |
| `binding` | True | Generate `__doc__`, `__module__` etc. for `def` functions |

## Our Standard Header

Every `.pyx` file in this project starts with:
```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
```

## Patterns

### File-Wide Header Format

The header comment must be a single line with comma-separated `key=value` pairs. Newlines or extra content in the directive line cause parsing failures.

```cython
# BAD — directive line contains code after the value
# cython: cdivision=True
# ^^^ works, but this does NOT:
# cython: cdivision=True\n\ndef my_func():  ...
# ERROR: "cdivision directive must be set to True or False, got 'True\n\ndef...'"

# BAD — directive value has trailing content
# cython: wraparound=False\n\ncimport cython\n\ncpdef tuple lz77_...
# ERROR: "wraparound directive must be set to True or False, got 'False\n\ncimport...'"

# GOOD — clean single-line header
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
```

This error appeared in traces across cryptography and compression categories when models embedded code in the directive line.

### Per-Function Decorators vs File-Wide

Use per-function decorators when only some functions need unsafe optimizations:

```cython
import cython

# Safe default for the file
def process_input(list data):
    # boundscheck is ON here — good for user-facing code
    ...

@cython.boundscheck(False)
@cython.wraparound(False)
def _hot_inner_loop(double[::1] arr):
    # boundscheck OFF here — performance-critical
    cdef int i
    for i in range(arr.shape[0]):
        arr[i] *= 2.0
```

### cdivision Safety

`cdivision=True` gives C-style integer division (truncates toward zero, no ZeroDivisionError). This is faster but changes semantics:

```cython
# With cdivision=True:
#   -7 // 2 == -3  (C truncation toward zero)
#   7 // 0  → undefined behavior (no exception!)
#
# With cdivision=False (Python default):
#   -7 // 2 == -4  (Python floor division)
#   7 // 0  → ZeroDivisionError

# Safe pattern: cdivision=True file-wide, but guard divisions
# cython: cdivision=True
cdef double safe_divide(double a, double b):
    if b == 0.0:
        return 0.0    # or raise, or return inf
    return a / b
```

### initializedcheck and Memoryviews

`initializedcheck=True` (default) checks that a memoryview is initialized before access. Disable for performance when you know the memoryview is always set:

```cython
@cython.initializedcheck(False)
@cython.boundscheck(False)
def fill_buffer(double[::1] buf, double value):
    cdef int i
    for i in range(buf.shape[0]):
        buf[i] = value
```

### Non-Existent Directives

Traces show models inventing directives that don't exist:

```cython
# BAD — these are NOT real Cython directives
# cython: cython.parallel.atomic  → ERROR: "No such directive: cython.parallel.atomic"
# cython: reduction               → not a directive
# BAD — invalid prange keyword
for i in prange(n, nogil=True, reduction='+'):  # ERROR: "Invalid keyword argument: reduction"
    total += arr[i]

# GOOD — prange auto-detects += as reduction
for i in prange(n, nogil=True):
    total += arr[i]    # += automatically treated as reduction
```

The `reduction` keyword appeared 22 times in numerical traces. Cython's prange handles reductions implicitly via `+=`, `-=`, `*=` operators.

## Directive Combinations for Maximum Performance

Typical high-performance combination used in traces achieving >1000x speedup:

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: initializedcheck=False
from cython.parallel cimport prange

def fast_computation(double[::1] data):
    cdef int i, n = data.shape[0]
    cdef double total = 0.0
    for i in range(n):          # or prange(n, nogil=True) if appropriate
        total += data[i] * data[i]
    return total
```

Average annotation scores across 2,795 traces: **0.846–0.898**, indicating most optimized code achieves >85% pure-C lines.

## Trace Statistics

Across ~2,800 traces from 6 categories:

| Error Pattern | Count | Categories |
|--------------|-------|------------|
| cdivision/wraparound parse failure | 3+ | cryptography, compression, dynamic_programming |
| "Invalid keyword argument: reduction" | 22+ | numerical |
| "No such directive: cython.parallel.atomic" | 4 | numerical |
| "'exceptval_check' not a valid cython attribute" | 2 | compression |

## Gotchas

1. **One-line header** — The `# cython:` directive must be self-contained on one line. No newlines in the value.
2. **No `reduction` keyword** — prange auto-detects `+=`, `*=` etc. Don't pass `reduction=` argument.
3. **cdivision + zero** — With `cdivision=True`, dividing by zero is undefined behavior, not an exception.
4. **boundscheck scope** — File-wide `boundscheck=False` applies everywhere, including user-input-facing code. Prefer per-function decorators.
5. **overflowcheck cost** — `overflowcheck=True` adds overhead to every integer operation. Use sparingly.

## See Also

[[optimization]], [[typing]], [[parallelism]], [[pitfalls]]
