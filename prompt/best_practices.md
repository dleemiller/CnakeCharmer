# Cython Best Practices Guide       

This comprehensive guide covers the essential aspects of writing efficient, maintainable Cython code. It presents both pure Python syntax and Cython-specific syntax side by side, offering guidance on when to use each approach.

## Table of Contents

- [Fundamentals](#fundamentals)
- [Static Typing](#static-typing)
- [Function Types](#function-types)
- [Memory Management](#memory-management)
- [NumPy Integration](#numpy-integration)
- [Performance Optimization](#performance-optimization)
- [Parallelization](#parallelization)
- [Error Handling](#error-handling)
- [C Interoperability](#c-interoperability)
- [Style Guide](#style-guide)

## Fundamentals

### Core Concepts

Cython is a compiler that generates optimized C code from Python-like syntax, providing the expressiveness of Python with C's performance. It bridges Python and C, allowing you to gradually add static typing and C-level optimizations to your Python code.

### Syntax Options

Cython offers two ways to write code:

1. **Cython-specific syntax** (.pyx files):
   - Direct C integration using `cdef` and `cpdef`
   - More concise for heavily typed code
   - Requires compilation step

2. **Pure Python syntax** (.py files):
   - Uses Python type hints and `cimport cython`
   - Compatible with standard Python tools
   - Enables gradual optimization
   - Better for maintainability

### Compilation

Compile your Cython code using `cythonize`:

```bash
# Simple compilation
cythonize -i mymodule.pyx

# With OpenMP support for parallelization
cythonize -i mymodule.pyx --compile-args=-fopenmp
```

Or add it to your `setup.py`:

```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="My package",
    ext_modules=cythonize("mymodule.pyx"),
)
```

## Static Typing

Static typing is the most important feature for performance gains in Cython.

### Basic Types Declaration

#### Pure Python Syntax

```python
import cython

def func():
    # Declare C types using annotations
    value: cython.int = 42
    name: cython.p_char = b"hello"
    coords: cython.double[3] = [1.0, 2.0, 3.0]
```

#### Cython Syntax

```cython
def func():
    # Declare C types using cdef
    cdef int value = 42
    cdef char* name = "hello"
    cdef double coords[3]
    coords = [1.0, 2.0, 3.0]
```

### Structs and Unions

#### Pure Python Syntax

```python
import cython

# Define a struct
Point = cython.struct(x=cython.int, y=cython.int, packed=True)

def compute_distance(p1: Point, p2: Point) -> cython.double:
    dx: cython.double = p1.x - p2.x
    dy: cython.double = p1.y - p2.y
    return cython.sqrt(dx*dx + dy*dy)
```

#### Cython Syntax

```cython
# Define a packed struct to eliminate padding
cdef packed struct Point:
    int x
    int y

def compute_distance(Point p1, Point p2):
    cdef double dx = p1.x - p2.x
    cdef double dy = p1.y - p2.y
    return sqrt(dx*dx + dy*dy)
```

## Function Types

Cython provides three types of functions, each offering different performance characteristics and compatibility with Python.

### `def` Functions (Python)

Use for functions that need to be callable from Python.

```python
def calculate_sum(numbers):
    """Regular Python function - slowest but most flexible."""
    total = 0
    for num in numbers:
        total += num
    return total
```

### `cdef` Functions (C-only)

Use for internal functions that need maximum performance.

#### Pure Python Syntax

```python
import cython

@cython.cfunc
def _c_fast_sum(numbers: cython.double[:], size: cython.int) -> cython.double:
    """C-only function - fastest but not callable from Python."""
    total: cython.double = 0.0
    for i in range(size):
        total += numbers[i]
    return total
```

#### Cython Syntax

```cython
cdef double _c_fast_sum(double[:] numbers, int size):
    """C-only function - fastest but not callable from Python."""
    cdef double total = 0.0
    cdef int i
    for i in range(size):
        total += numbers[i]
    return total
```

### `cpdef` Functions (Hybrid)

Use when you need both C-level performance and Python accessibility.

#### Pure Python Syntax

```python
import cython

@cython.ccall
def fast_sum(numbers: cython.double[:]) -> cython.double:
    """Hybrid function - fast and Python-callable."""
    return _c_fast_sum(numbers, len(numbers))
```

#### Cython Syntax

```cython
cpdef double fast_sum(double[:] numbers):
    """Hybrid function - fast and Python-callable."""
    return _c_fast_sum(numbers, len(numbers))
```

## Memory Management

Memory management is critical in Cython. Proper allocation and deallocation prevent memory leaks and crashes.

### Manual Memory Allocation

#### Pure Python Syntax

```python
import cython
from libc.stdlib cimport malloc, free

def allocate_array(size: cython.int) -> cython.double[:]:
    """Allocate a C array and return it as a memoryview."""
    # Allocate memory for the array
    ptr = cython.cast(cython.p_double, malloc(size * cython.sizeof(cython.double)))
    
    # Check if allocation was successful
    if not ptr:
        raise MemoryError("Failed to allocate memory")
    
    # Create a memoryview from the pointer
    # The try-finally ensures memory is freed if an exception occurs
    try:
        # Initialize with zeros
        for i in range(size):
            ptr[i] = 0.0
            
        # Create a memoryview that owns the pointer
        result = cython.cast(cython.double[:size], ptr)
        return result
    except:
        # Clean up on error
        free(ptr)
        raise
```

#### Cython Syntax

```cython
from libc.stdlib cimport malloc, free

def allocate_array(int size):
    """Allocate a C array and return it as a memoryview."""
    # Allocate memory for the array
    cdef double* ptr = <double*>malloc(size * sizeof(double))
    
    # Check if allocation was successful
    if ptr == NULL:
        raise MemoryError("Failed to allocate memory")
    
    # Create a memoryview from the pointer
    # The try-finally ensures memory is freed if an exception occurs
    try:
        # Initialize with zeros
        cdef int i
        for i in range(size):
            ptr[i] = 0.0
            
        # Create a memoryview that owns the pointer
        cdef double[:] result = <double[:size]>ptr
        return result
    except:
        # Clean up on error
        free(ptr)
        raise
```

### Extension Types with Resource Management

Extension types provide an object-oriented way to manage C resources.

#### Pure Python Syntax

```python
import cython
from libc.stdlib cimport malloc, free

@cython.cclass
class Array:
    """A simple wrapper around a C array with proper cleanup."""
    
     cython.p_double
    size: cython.int
    
    def __cinit__(self, size: cython.int):
        """Constructor - allocates memory."""
        self.size = size
        self.data = cython.cast(cython.p_double, malloc(size * cython.sizeof(cython.double)))
        if not self.
            raise MemoryError("Failed to allocate memory")
        
        # Initialize with zeros
        for i in range(size):
            self.data[i] = 0.0
    
    def __dealloc__(self):
        """Destructor - ensures memory is freed."""
        if self.data != NULL:
            free(self.data)
            self.data = NULL
    
    def __getitem__(self, idx: cython.int) -> cython.double:
        """Access elements by index."""
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of bounds")
        return self.data[idx]
    
    def __setitem__(self, idx: cython.int, value: cython.double):
        """Set elements by index."""
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of bounds")
        self.data[idx] = value
```

#### Cython Syntax

```cython
from libc.stdlib cimport malloc, free

cdef class Array:
    """A simple wrapper around a C array with proper cleanup."""
    
    cdef double* data
    cdef int size
    
    def __cinit__(self, int size):
        """Constructor - allocates memory."""
        self.size = size
        self.data = <double*>malloc(size * sizeof(double))
        if self.data == NULL:
            raise MemoryError("Failed to allocate memory")
        
        # Initialize with zeros
        cdef int i
        for i in range(size):
            self.data[i] = 0.0
    
    def __dealloc__(self):
        """Destructor - ensures memory is freed."""
        if self.data != NULL:
            free(self.data)
            self.data = NULL
    
    def __getitem__(self, int idx):
        """Access elements by index."""
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of bounds")
        return self.data[idx]
    
    def __setitem__(self, int idx, double value):
        """Set elements by index."""
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of bounds")
        self.data[idx] = value
```

## NumPy Integration

Memoryviews provide fast, direct access to NumPy arrays without the Python overhead.

### Using Memoryviews

#### Pure Python Syntax

```python
import cython
import numpy as np

def matrix_multiply(a: cython.double[:, :], b: cython.double[:, :]) -> cython.double[:, :]:
    """Multiply two matrices using memoryviews for efficient access."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    
    m: cython.int = a.shape[0]
    n: cython.int = a.shape[1]
    p: cython.int = b.shape[1]
    
    # Create output matrix
    result = np.zeros((m, p), dtype=np.double)
    result_view: cython.double[:, :] = result
    
    # Perform multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result_view[i, j] += a[i, k] * b[k, j]
    
    return result_view
```

#### Cython Syntax

```cython
import numpy as np

def matrix_multiply(double[:, :] a, double[:, :] b):
    """Multiply two matrices using memoryviews for efficient access."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int p = b.shape[1]
    
    # Create output matrix
    result = np.zeros((m, p), dtype=np.double)
    cdef double[:, :] result_view = result
    
    cdef int i, j, k
    # Perform multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result_view[i, j] += a[i, k] * b[k, j]
    
    return result_view
```

### Optimized Contiguous Memory Access

#### Pure Python Syntax

```python
import cython
import numpy as np

@cython.boundscheck(False)  # Disable bounds checking
@cython.wraparound(False)   # Disable negative indexing
def fast_dot_product(a: cython.double[::1], b: cython.double[::1]) -> cython.double:
    """Compute dot product of two vectors with contiguous memory."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must have the same length")
    
    n: cython.int = a.shape[0]
    result: cython.double = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
    
    return result
```

#### Cython Syntax

```cython
import numpy as np

@cython.boundscheck(False)  # Disable bounds checking
@cython.wraparound(False)   # Disable negative indexing
def fast_dot_product(double[::1] a, double[::1] b):
    """Compute dot product of two vectors with contiguous memory."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must have the same length")
    
    cdef int n = a.shape[0]
    cdef double result = 0.0
    cdef int i
    
    for i in range(n):
        result += a[i] * b[i]
    
    return result
```

## Performance Optimization

### Loop Optimization

#### Pure Python Syntax

```python
import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def optimize_loops( cython.double[:, :]) -> cython.double:
    """Example of optimized loop patterns."""
    # Extract loop invariants
    height: cython.int = data.shape[0]
    width: cython.int = data.shape[1]
    
    # Pre-compute constants
    scale: cython.double = 2.0
    threshold: cython.double = 100.0
    
    total: cython.double = 0.0
    
    # Type loop indices for faster iteration
    i: cython.int
    j: cython.int
    
    # Use C-level for loops
    for i in range(height):
        # Hoist inner loop invariants
        row_offset: cython.double = i * scale
        
        for j in range(width):
            # Store repeated calculations
            value: cython.double = data[i, j]
            scaled_value: cython.double = value * scale + row_offset
            
            # Add to total if below threshold
            if scaled_value < threshold:
                total += scaled_value
    
    return total
```

#### Cython Syntax

```cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def optimize_loops(double[:, :] data):
    """Example of optimized loop patterns."""
    # Extract loop invariants
    cdef int height = data.shape[0]
    cdef int width = data.shape[1]
    
    # Pre-compute constants
    cdef double scale = 2.0
    cdef double threshold = 100.0
    
    cdef double total = 0.0
    
    # Type loop indices for faster iteration
    cdef int i, j
    cdef double row_offset, value, scaled_value
    
    # Use C-level for loops
    for i in range(height):
        # Hoist inner loop invariants
        row_offset = i * scale
        
        for j in range(width):
            # Store repeated calculations
            value = data[i, j]
            scaled_value = value * scale + row_offset
            
            # Add to total if below threshold
            if scaled_value < threshold:
                total += scaled_value
    
    return total
```

### Fused Types (Templates)

#### Pure Python Syntax

```python
import cython
import numpy as np

# Define a fused type that works with multiple numeric types
numeric = cython.fused_type(cython.int, cython.float, cython.double)

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_array( numeric[:]) -> numeric:
    """Sum array elements with type specialized for performance."""
    n: cython.int = data.shape[0]
    total: numeric = 0
    
    for i in range(n):
        total += data[i]
    
    return total
```

#### Cython Syntax

```cython
import numpy as np

# Define a fused type that works with multiple numeric types
ctypedef fused numeric:
    int
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_array(numeric[:] data):
    """Sum array elements with type specialized for performance."""
    cdef int n = data.shape[0]
    cdef numeric total = 0
    cdef int i
    
    for i in range(n):
        total += data[i]
    
    return total
```

## Parallelization

### Using prange for Parallel Loops

#### Pure Python Syntax

```python
import cython
from cython.parallel import prange
import numpy as np

# Ensure you compile with OpenMP support:
# cythonize -i mymodule.py --compile-args=-fopenmp --link-args=-fopenmp

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_sum(data: cython.double[:, :]) -> cython.double:
    """Sum elements of a 2D array in parallel."""
    height: cython.int = data.shape[0]
    width: cython.int = data.shape[1]
    
    total: cython.double = 0.0
    
    # The outer loop is parallelized with OpenMP
    # Use nogil since we're only working with C types
    for i in prange(height, nogil=True, num_threads=4):
        row_sum: cython.double = 0.0
        
        for j in range(width):
            row_sum += data[i, j]
            
        # Using with gil: block to safely update shared variable
        with cython.gil:
            total += row_sum
    
    return total
```

#### Cython Syntax

```cython
from cython.parallel import prange
import numpy as np

# Ensure you compile with OpenMP support:
# cythonize -i mymodule.pyx --compile-args=-fopenmp --link-args=-fopenmp

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_sum(double[:, :] data):
    """Sum elements of a 2D array in parallel."""
    cdef int height = data.shape[0]
    cdef int width = data.shape[1]
    
    cdef double total = 0.0
    cdef double row_sum
    cdef int i, j
    
    # The outer loop is parallelized with OpenMP
    # Use nogil since we're only working with C types
    for i in prange(height, nogil=True, num_threads=4):
        row_sum = 0.0
        
        for j in range(width):
            row_sum += data[i, j]
            
        # Using with gil: block to safely update shared variable
        with gil:
            total += row_sum
    
    return total
```

## Error Handling

### Handling Exceptions in Cython Functions

#### Pure Python Syntax

```python
import cython

@cython.cfunc
@cython.exceptval(-1, check=True)
def safe_divide(a: cython.int, b: cython.int) -> cython.int:
    """C function that can raise Python exceptions."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a // b

def divide_values(values: list, divisor: int) -> list:
    """Safely divide a list of values and handle errors."""
    result = []
    
    for i, value in enumerate(values):
        try:
            result.append(safe_divide(value, divisor))
        except ZeroDivisionError:
            print(f"Warning: Cannot divide {value} by zero at index {i}")
            result.append(0)
    
    return result
```

#### Cython Syntax

```cython
cdef int safe_divide(int a, int b) except? -1:
    """C function that can raise Python exceptions."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a // b

def divide_values(list values, int divisor):
    """Safely divide a list of values and handle errors."""
    result = []
    
    for i, value in enumerate(values):
        try:
            result.append(safe_divide(value, divisor))
        except ZeroDivisionError:
            print(f"Warning: Cannot divide {value} by zero at index {i}")
            result.append(0)
    
    return result
```

## C Interoperability

### Wrapping C Functions

#### C Header (math_utils.h)

```c
/* math_utils.h */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

/* Calculate the factorial of a number */
int factorial(int n);

/* Calculate the GCD of two numbers */
int gcd(int a, int b);

#endif
```

#### Cython Definition File (math_utils.pxd)

```cython
# math_utils.pxd
cdef extern from "math_utils.h":
    int factorial(int n)
    int gcd(int a, int b)
```

#### Python Wrapper (Pure Python Syntax)

```python
# math_utils_wrapper.py
import cython
from cython.cimports.math_utils import factorial, gcd

def py_factorial(n: int) -> int:
    """Python-callable wrapper for the C factorial function."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial requires a non-negative integer")
    return factorial(n)

def py_gcd(a: int, b: int) -> int:
    """Python-callable wrapper for the C GCD function."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("GCD requires integer arguments")
    return gcd(a, b)
```

#### Python Wrapper (Cython Syntax)

```cython
# math_utils_wrapper.pyx
from math_utils cimport factorial, gcd

def py_factorial(n):
    """Python-callable wrapper for the C factorial function."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial requires a non-negative integer")
    return factorial(n)

def py_gcd(a, b):
    """Python-callable wrapper for the C GCD function."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("GCD requires integer arguments")
    return gcd(a, b)
```

## Style Guide

### Naming Conventions

- `_c_my_function()`: Use underscore prefix for C-only functions (`cdef`)
- `variable_s`: Suffix with `_s` for C structs
- `data_p`: Suffix with `_p` for C pointer variables
- `PyObject`: Prefix with `Py` for objects returned from pure C code

### Code Organization

- Separate implementation (`.pyx`/`.py`) and definition (`.pxd`) files
- Group related functionality into logical modules
- Keep C-level details in `.pxd` files for reuse
- Document public interfaces clearly

### Example of a Well-Structured Cython Module

#### Definition File (vector.pxd)

```cython
# vector.pxd
cdef struct Vector2D_s:
    double x
    double y

cdef class Vector2D:
    cdef Vector2D_s _v
    cdef double _magnitude
    cdef bint _magnitude_valid
    
    cpdef double get_magnitude(self)
    cpdef Vector2D normalize(self)
```

#### Implementation File (Cython Syntax)

```cython
# vector.pyx
from libc.math cimport sqrt

cdef class Vector2D:
    def __cinit__(self, double x=0.0, double y=0.0):
        self._v.x = x
        self._v.y = y
        self._magnitude_valid = False
    
    property x:
        def __get__(self):
            return self._v.x
        def __set__(self, double value):
            self._v.x = value
            self._magnitude_valid = False
    
    property y:
        def __get__(self):
            return self._v.y
        def __set__(self, double value):
            self._v.y = value
            self._magnitude_valid = False
    
    cpdef double get_magnitude(self):
        if not self._magnitude_valid:
            self._magnitude = sqrt(self._v.x * self._v.x + self._v.y * self._v.y)
            self._magnitude_valid = True
        return self._magnitude
    
    cpdef Vector2D normalize(self):
        cdef double mag = self.get_magnitude()
        if mag > 0:
            return Vector2D(self._v.x / mag, self._v.y / mag)
        return Vector2D()
    
    def __add__(self, other):
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self._v.x + (<Vector2D>other)._v.x, 
                        self._v.y + (<Vector2D>other)._v.y)
```

#### Implementation File (Pure Python Syntax)

```python
# vector.py
import cython
from libc.math cimport sqrt

@cython.cclass
class Vector2D:
    _v: Vector2D_s
    _magnitude: cython.double
    _magnitude_valid: cython.bint
    
    def __cinit__(self, x: cython.double = 0.0, y: cython.double = 0.0):
        self._v.x = x
        self._v.y = y
        self._magnitude_valid = False
    
    @property
    def x(self) -> cython.double:
        return self._v.x
    
    @x.setter
    def x(self, value: cython.double):
        self._v.x = value
        self._magnitude_valid = False
    
    @property
    def y(self) -> cython.double:
        return self._v.y
    
    @y.setter
    def y(self, value: cython.double):
        self._v.y = value
        self._magnitude_valid = False
    
    @cython.ccall
    def get_magnitude(self) -> cython.double:
        if not self._magnitude_valid:
            self._magnitude = sqrt(self._v.x * self._v.x + self._v.y * self._v.y)
            self._magnitude_valid = True
        return self._magnitude
    
    @cython.ccall
    def normalize(self) -> 'Vector2D':
        mag: cython.double = self.get_magnitude()
        if mag > 0:
            return Vector2D(self._v.x / mag, self._v.y / mag)
        return Vector2D()
    
    def __add__(self, other):
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self._v.x + cython.cast(Vector2D, other)._v.x, 
                        self._v.y + cython.cast(Vector2D, other)._v.y)
```

By following these best practices, you'll write Cython code that is efficient, maintainable, and makes appropriate use of both Python and C features.
