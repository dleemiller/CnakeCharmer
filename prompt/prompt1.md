# Cython Programming Guide

This comprehensive guide covers Cython best practices, performance vs Python compatibility trade-offs, and appropriate use of both Cython and Pure Python syntax. The guide prioritizes clarity, efficiency, and safety.

## I. Fundamental Principles

- **Cython's Core:** Cython bridges Python and C, compiling Python-like code to optimized C and boosting performance through static typing and direct C API interaction.

- **Syntax Options:**
  - **.pyx (Cython Syntax):** Uses `cdef` and `cpdef` for direct C integration. Fast, concise, but less Python-compatible. Ideal for integrating with C structures and operations.
  
  - **.py (Pure Python Syntax):** Leverages Python type hints (PEP 484, PEP 526) with `cimport cython`. More Python-compatible and readable, allows gradual optimization, and works with standard Python tools.

- **Compilation:** Use `cythonize` or `setup.py` (`build_ext --inplace`). Enable OpenMP (e.g., `-fopenmp`) for parallelization. Always profile *before* optimizing. Use annotation with `-a` to generate HTML output highlighting Python interactions.

- **Incremental Optimization:** Start with valid Python syntax in `.py` files and gradually add C code where performance is needed.

## II. Static Typing & Data Structures

- **`cdef` (Cython):** Declares C-level variables and functions. Only used in `.pyx` or `.pxd` files.

- **Type Hints (Pure Python):** 
  ```python
  import cython
  
  x: cython.int = 1
  
  @cython.cfunc
  def add_one(x: cython.int) -> cython.int:
      return x + 1
  ```
  You *must* `cimport cython`. Use standard Python typing hints alongside Cython type hints.

- **Data Types:** Prefer C types (`int`, `float`, `double`, `char`, `size_t`, pointers) for performance. Python objects (`list`, `dict`, classes) are usable *with performance caveats*. Ctuples are efficient alternatives to Python tuples. `cdef packed struct` removes padding for memory efficiency.

## III. Function Definitions and Choosing the Right Type

- **`def` (Python):** Standard Python function. Slowest but fully Python-callable.

- **`cdef` (C):** C-only function. Fastest but *not* Python-callable and cannot perform error handling by default.

- **`cpdef` (Hybrid):** Creates both C and Python entry points with small virtual function table overhead. Can be called from both languages for better flexibility.

- **Choosing Function Types:**
  - Use `def` when Python compatibility is needed
  - Use `cdef` for Cython-only contexts where speed is paramount
  - Use `cpdef` when balancing performance and accessibility

- Always provide clear return type declarations, especially for primitives (non-Python objects), to optimize performance.

## IV. Memory Management – The Most Critical Aspect!

- **Manual Allocation (C style):**
  ```cython
  from libc.stdlib cimport malloc, free
  
  cdef int* data_p = <int*>malloc(10 * sizeof(int))
  if not data_p:
      raise MemoryError("Failed to allocate memory")
  try:
      # Use data_p
      for i in range(10):
          data_p[i] = i
  finally:
      free(data_p)
      data_p = NULL  # Prevent dangling pointer
  ```

- **Python Heap Allocation:**
  ```cython
  from cpython.mem cimport PyMem_Malloc, PyMem_Free
  
  cdef int* data_p = <int*>PyMem_Malloc(10 * sizeof(int))
  if not data_p:
      raise MemoryError("Failed to allocate memory")
  try:
      # Use data_p
      for i in range(10):
          data_p[i] = i
  finally:
      PyMem_Free(data_p)
      data_p = NULL
  ```

- **Extension Types:** Allocate in `__cinit__`, deallocate in `__dealloc__` (for `cdef class` or `@cclass`).

- **Resource Acquisition Is Initialization (RAII):** Use context managers (`with` statement) for automatic resource cleanup.

- **Ownership:** *Clearly define memory ownership to prevent double frees and dangling pointers*. Set pointers to `NULL` after freeing. Enclose allocation/deallocation in `try...finally` to guarantee cleanup.

## V. NumPy Integration (Modern Approach Preferred)

- **Legacy Buffer Protocol (Avoid):** `cnp.ndarray[DTYPE_t, ndim=2]`.

- **Memoryviews (Preferred):**
  ```cython
  # Cython syntax
  cdef double[:, :] matrix
  
  # Pure Python syntax
  matrix: cython.double[:, :]
  ```

- **Contiguity:**
  ```cython
  # C-contiguous (optimized for row access)
  cdef double[:, ::1] c_matrix
  
  # Fortran-contiguous (optimized for column access)
  cdef double[::1, :] f_matrix
  ```
  Assigning non-contiguous buffers raises `ValueError`. `cython.view` includes memory layout options (generic, strided, etc.).

- **ufuncs:** Use `@cython.ufunc` for element-wise array operations with scalar return types. Fused types support multiple return types. Use `nogil` for multithreading.

## VI. Performance Optimization – After Profiling

- **Profiling:** *Essential*. Use annotations (`-a`) to identify Python bottlenecks (shown as white lines in HTML output).

- **Loop Optimization:**
  ```cython
  # Optimized loop example
  cdef int i, n = len(array)  # Move invariant calculation outside
  cdef double total = 0.0
  
  for i in range(n):  # Type the loop index
      total += array[i]
  ```
  - Move loop-invariant calculations outside
  - Type loop indices (`cdef int i`)
  - Precompute repeated expressions
  - Consider manually unrolling tiny loops
  - Prioritize static typing in innermost loops

- **Optimization Directives:**
  ```cython
  # File-level directives
  # cython: boundscheck=False, wraparound=False, cdivision=True
  
  # Function-level directives
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def fast_function():
      # ...
  ```
  Use judiciously after careful testing and safety considerations.

- **Fused Types:**
  ```cython
  # Cython syntax
  ctypedef fused numeric:
      int
      double
  
  cdef numeric add_one(numeric x):
      return x + 1
  
  # Pure Python syntax
  numeric = cython.fused_type(cython.int, cython.double)
  
  @cython.cfunc
  def add_one(x: numeric) -> numeric:
      return x + 1
  ```
  Templates for multiple C types. A function should have *one* fused-type argument.

- **Inline Functions:**
  ```cython
  cdef inline int min_val(int a, int b) nogil:
      return a if a <= b else b
  ```
  Eliminate function-call overhead for small, frequently called functions.

## VII. Parallelization (with GIL Release)

```cython
from cython.parallel import prange
import numpy as np

def parallel_sum(double[:] arr):
    cdef:
        int i
        double total = 0.0
        int n = arr.shape[0]
    
    # Release GIL for true parallelism
    for i in prange(n, nogil=True, num_threads=4):
        total += arr[i]
    
    return total
```

- Release the GIL to enable true parallelization
- Use `from cython.parallel import prange`
- Set `nogil=True` with `prange`
- Use `with nogil:` context or function clause for code blocks that must be GIL-safe
- Use with reduction operations to accumulate values
- Enable OpenMP during compilation (`-fopenmp` flag)

## VIII. Error Handling

```cython
# Python exceptions
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return float('inf')
    finally:
        # Clean-up code here
        pass

# C errors in cdef functions
cdef int divide(int a, int b) except? -1:
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a // b

# Functions that should never raise exceptions
cdef int add(int a, int b) noexcept:
    return a + b
```

- **Python Exceptions:** Use `try...except...finally`
- **C Errors:** Check return codes; map errors to Python exceptions
- **`noexcept`:** Indicates a function should *never* throw exceptions
- In `cdef` functions, handle exceptions from C values using `except + ExceptionType`

## IX. C Interoperability

```cython
# In mymodule.pxd
cdef extern from "math.h":
    double sqrt(double x)
    double pow(double x, double y)

# In mymodule.pyx
cdef double calculate_hypotenuse(double a, double b):
    return sqrt(pow(a, 2) + pow(b, 2))

# Python-accessible wrapper
def hypotenuse(a, b):
    return calculate_hypotenuse(a, b)
```

- Use `cdef extern from "header.h":` to declare C functions, variables, and structs
- Use `.pxd` files for declarations and add named parameters for keyword support
- *Memory management is CRITICAL* - allocate and deallocate correctly
- Check `Cython/Includes` for existing `.pxd` files for standard C functions

## X. Style Guide and Standards

- `_c_my_function()`: Prefix with underscore for C-only functions (`cdef`)
- `point_s`: Suffix with `_s` for C structs
- `data_p`: Suffix with `_p` for C pointer variables
- `Py`: Prefix for objects returned from pure C code

## XI. Type Hint Consistency

```python
# Consistent type hints between Python and Cython
import cython
from typing import List

def process_data(values: List[float]) -> cython.int:
    result: cython.int = 0
    for v in values:
        result += int(v)
    return result
```

- Ensure consistency between Python hints and Cython types
- `int` maps to `cython.int`, etc.
- Review Cython's type conversion rules for proper mapping

## XII. Extension Types (cdef classes/cClasses)

```cython
# Cython syntax
cdef class Rectangle:
    cdef:
        int width
        int height
        readonly double area  # Python-readable
        public str name       # Python-readable/writable
    
    def __cinit__(self, int width, int height, str name="Rectangle"):
        self.width = width
        self.height = height
        self.name = name
        self.area = width * height
    
    def __dealloc__(self):
        # Clean up any manually allocated resources
        pass

# Pure Python syntax
@cython.cclass
class Rectangle:
    width: cython.int
    height: cython.int
    area: cython.double
    name: str
    
    def __init__(self, width: cython.int, height: cython.int, name: str = "Rectangle"):
        self.width = width
        self.height = height
        self.name = name
        self.area = width * height
```

- Store data in C struct instead of Python dictionary for lower overhead
- Include `__cinit__` (constructor) and `__dealloc__` (destructor) for proper resource management
- Declare using `cdef class` or the Pure Python `@cython.cclass` annotation
- Pre-declare attributes with C types using `cdef` or annotations
- Use `public`, `readonly`, or private (default) access modifiers

## XIII. Structs and Unions

```cython
# Cython syntax for struct
cdef packed struct Point:
    int x
    int y

# Using the struct
cdef Point p
p.x = 10
p.y = 20

# Pure Python syntax
Point = cython.struct(x=cython.int, y=cython.int, packed=True)
p: Point = Point()
p.x = 10
p.y = 20
```

- Package data into compact memory blocks
- Use `cdef struct` (Cython syntax) or `cython.struct` (Pure Python syntax)
- Add `packed` keyword to create structs without padding for smaller size
- Access from Python through public members, optionally with `readonly` modifier

## XIV. Naming Patterns (`.pxd` Files)

```cython
# In my_module.pxd
cdef int _c_calculate_area(int width, int height)

# In my_module.pyx
cdef int _c_calculate_area(int width, int height):
    return width * height

# Python-accessible wrapper
def calculate_area(width, height):
    return _c_calculate_area(width, height)
```

- Implementation and declaration names must match exactly
- Follow consistent naming conventions for different types of entities
- Use `.pxd` files for sharing declarations between modules

## Best Practices for Code Examples

When providing code examples, always:
1. Specify file type (`.pyx` or `.py`) and include `cimport cython` and directives for `.py` files
2. Include necessary `import` statements
3. Explain each line, especially type declarations and memory management
4. Highlight performance considerations and trade-offs
5. Justify why pure Python versus Cython syntax is used
6. Show declarations using `.pxd` files when implementing external C functions
7. Demonstrate correct memory management practices
8. Prioritize memory safety, speed, and simplicity
