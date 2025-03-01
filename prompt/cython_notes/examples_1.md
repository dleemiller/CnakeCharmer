# Canonical Examples: Python vs. Cython

## Overview

This guide provides a set of canonical examples comparing implementations in pure Python versus Cython, with a focus on Cython 3 best practices. The examples showcase different language features and usage scenarios, highlighting the performance improvements gained with Cython's static typing and direct C code generation.

Each example includes:
1. A pure Python implementation
2. The equivalent Cython-optimized code

> **Note:** These examples demonstrate core concepts of Cython. Actual performance gains will vary based on your specific use case, hardware, and compiler optimizations. Always profile before optimizing.

## Example 1: Basic Arithmetic

**Aim**: Demonstrate performance differences in basic arithmetic operations.

### Python

```python
def py_add(a, b):
    return a + b
```

### Cython

```python
def add(int a, int b):
    return a + b
```

**Performance Comparison**: Adding static types to a simple addition provides a small but measurable speedup by removing interpreter overhead.

## Example 2: Looping with Range

**Aim**: Showcase the impact of static typing on loop performance.

### Python

```python
def py_sum_range(n):
    s = 0
    for i in range(n):
        s += i
    return s
```

### Cython

```python
def sum_range(int n):
    cdef int i, s = 0
    for i in range(n):
        s += i
    return s
```

**Performance Comparison**: With both index and result variables defined as C types, loop speed improves significantly. Larger loops show greater speedups.

## Example 3: Array Processing with NumPy

**Aim**: Demonstrate efficient array manipulation using memoryviews.

### Python

```python
import numpy as np

def py_array_sum(arr):
    sum_val = 0
    n = arr.shape[0]
    for i in range(n):
        sum_val += arr[i]
    return sum_val
```

### Cython

```python
import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def array_sum(double[:] arr):  # Modern memoryview syntax
    cdef double sum_val = 0
    cdef int i
    
    for i in range(arr.shape[0]):
        sum_val += arr[i]
    return sum_val
```

**Performance Comparison**: Using typed memoryviews (Cython 3 preferred approach) over NumPy arrays provides a significant performance boost for numerical operations. Disabling bounds checking further increases performance where it's safe to do so.

## Example 4: Struct Usage

**Aim**: Illustrate the performance benefits of using C structs for data grouping.

### Python

```python
def py_process_points(points):  # points is a list of (x, y) tuples
    result = 0
    for p in points:
        x, y = p
        result += x * y
    return result
```

### Cython

```python
# Define the C struct
cdef struct Point:
    int x
    int y

# Function using the struct
cdef int point_function(Point p):
    return p.x * p.y

def process_points(points):
    cdef int result = 0
    cdef Point p
    
    for point in points:
        p.x, p.y = point  # Unpack tuple into struct
        result += point_function(p)
    return result
```

**Performance Comparison**: Using C structures improves memory layout and provides direct data access, enhancing performance for calculations on complex data structures. The tradeoff is reduced flexibility compared to Python's dynamic typing.

## Example 5: Callbacks and C Integration

**Aim**: Illustrate the use of callback functions with C libraries.

First, declare the function in a C header:

```c
// In C header
typedef int (*callback_t)(int);
int call_with(int x, callback_t foo);
```

### Cython

```python
# Declare the external C function
cdef extern from "myclib.h":
    ctypedef int (*callback_t)(int)
    int call_with(int x, callback_t foo)

# Define a callback function
@cython.cfunc
def cy_callback(x: cython.int) -> cython.int:
    return x * x

# Function to call the C function with our callback
def call_c_func(int x):
    return call_with(x, cy_callback)
```

**Performance Comparison**: Cython 3's `@cython.cfunc` decorator creates true C function pointers, allowing for zero-overhead callbacks to C libraries.

## Example 6: Parallelization with OpenMP

**Aim**: Demonstrate Cython's support for parallel computing with OpenMP.

### Python

```python
import numpy as np

def py_vector_add(a, b):
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result
```

### Cython

```python
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def vector_add(double[:] a, double[:] b):
    cdef int i
    cdef int n = a.shape[0]
    cdef double[:] result = np.zeros(n, dtype=np.float64)
    
    # Parallel loop using OpenMP
    for i in prange(n, nogil=True):
        result[i] = a[i] + b[i]
        
    return np.asarray(result)
```

**Performance Comparison**: Using Cython's `prange` with the `nogil=True` option enables parallel execution across multiple CPU cores, providing significant speedups for computationally intensive operations.

## Conclusion

These examples demonstrate how Cython bridges Python and C, offering performance optimizations while maintaining code readability. Key Cython 3 best practices include:

1. Using typed memoryviews instead of NumPy arrays for maximum performance
2. Leveraging `@cython.cfunc` for creating true C function pointers
3. Taking advantage of parallel computing with OpenMP via `prange`
4. Using decorators like `@cython.boundscheck(False)` to disable unnecessary checks
5. Employing C structs for efficient data handling

By strategically adding static types and leveraging Cython's features, developers get the best of both worlds: Python's ease of use with C's performance.
