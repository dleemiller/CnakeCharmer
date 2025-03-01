**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 21: Pure Python Mode in Cython**

Pure Python mode allows Cython to leverage its benefits while adhering to standard Python syntax, enhancing code readability and compatibility.

**What It Is:**

*   A development style for Cython where you use only Python syntax and annotations for typing *instead* of custom Cython keywords like `cdef`. This enables gradual optimization and maintains readability.

**Enabling Pure Python Mode:**

*   You MUST use a `.py` file, **not** a `.pyx` file.
*   Add the following lines to the beginning of the file: This informs Cython to expect pure Python syntax and annotatation:

```python
# File: example.py (not .pyx)
# cython: language_level=3  # Ensure Python 3 syntax

import cython
```

**Core Differences: Syntax**

| Feature         | Traditional Cython (.pyx) | Pure Python Mode (.py) |
|-----------------|---------------------------|-------------------------|
| **Variable Declaration** | `cdef int x`             | `x: cython.int`       |
| **Function Declaration** | `cdef double func(...)`  | `@cython.cfunc def func(...) -> cython.double:` |

**Code Style Example:**

| Aspect                      | Cython Syntax                                  | Pure Python Syntax                                            |
|-----------------------------|------------------------------------------------|--------------------------------------------------------------|
| Import Statements            | `cimport numpy as np`                          | `from cython.cimports import numpy as np`                        |
| Variable Type Declarations    | `cdef int i`                                    | `i: cython.int`                                                 |
| Function Decorator     | `cdef double add(double a, double b):`                                |   `@cython.cfunc double add(a: cython.double, b: cython.double):`                                                    |
| Memory Allocations          | `cdef int* data = <int*>malloc(sizeof(int))`   | ` cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int), malloc(cython.sizeof(cython.int)))`                    |

**Common Function Decorators:**

*   These indicate Cython-specific behavior while maintaining valid Python code.

    *   `@cython.cfunc`:  Marks a function as a C-level function (similar to `cdef`).  Not callable directly from Python.
    *   `@cython.ccall`: Creates function that combines characteristics from both Python and C.
    *   `@cython.nogil`: Specifies that function releases the GIL (for thread-safe code).
    *   `@cython.exceptval(value, check=True)`:  Handles C exceptions; `value` is the error return value, and `check=True` enables exception checking.

**Code Example:**

```python
import cython

@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=True)
def compute(x: cython.int) -> cython.int:
    return x * x
```

**Extension Types (Classes) in Pure Python Mode:**

```python
import cython

@cython.cclass
class Point:
    x: cython.double  # C attribute
    y: cython.double

    def __init__(self, x: cython.double, y: cython.double):
        self.x = x
        self.y = y
```

**Memory Management (Pointers):**

*   Manual memory management (requires `malloc`, `free`):

```python
import cython
from libc.stdlib cimport malloc, free

buffer: cython.p_double = cython.cast(cython.p_double, malloc(100 * cython.sizeof(cython.double)))
# ...use buffer...
free(buffer)
```

**Trade-offs:**

| Advantage                                     | Disadvantage                                                                      |
|-----------------------------------------------|-----------------------------------------------------------------------------------|
| Standard Python Syntax                        | Slight runtime overhead vs. traditional Cython.                                |
| Gradual Optimization                          | verbose C-casts when allocating memory (avoid when you can)                        |
| Integration with Python development tools       | Pure mode is not as mature as Cython (.pyx); some edge cases may exists.           |
| Easy to read , modify by Python developers (as type hints can ignored).  | |

**When to Use Pure Python Mode:**

*   ✅  When you must have code that is valid Python.
*   ✅  If you already use type annotations in the standard Python library and must be fully implemented for C-API type checking.
*   ✅  Incremental optimization (start with valid Python, add types slowly).
*   ✅  Easier for Python-familiar contributors to read and modify the code.
*   ❌ If pure and simple high performance is needed
*   ❌ Working and porting complex C/C++ codes, better pick standard Cython

**Best Practices:**

* Consistently use type annotations or don't.
* Annotate with types from pure Python whenever possible.
* Combine Python for the main operations and small C/C++ files for the most time intensive code.
* Use standard syntax (and .py extensions) when you need to distribute code that is primarily used by other Python users and avoid requiring special tooling. 
* C-casts type definitions: The Cython type is explicitly type-casted to Cython pointer types and function to allocate ,copy, and resize with external C/C+ codes.
* The type-casting needs all cases to be evaluated with External call.

**Compilation Command:**

```bash
cythonize mymodule.py  # Compiles to C and builds the extension
python setup.py build_ext --inplace # Compiles to C and builds the extension
```

Pure Python mode is great for readability, compatibility, and incremental speedups but may not reach the absolute peak performance of finely tuned, traditional Cython.
