# Syntax Comparison: Pure Python vs. Cython

## Overview

Cython provides two main syntax variations for writing code:
1. **Pure Python syntax** - using type hints and decorators
2. **Cython syntax** - using `cdef` and other keywords

This guide highlights the key differences between these two approaches to help you choose the most appropriate style for your project.

### Key Considerations

When deciding which syntax to use, consider the following factors:

* **Readability**: Pure Python syntax allows `.py` files to remain valid Python code
* **Flexibility**: Pure Python enables easier refactoring without specific Cython knowledge
* **C/C++ Integration**: Cython syntax provides more explicit control over C/C++ features
* **Ease of Use**: Pure Python might be easier for beginners, while Cython syntax is generally more concise in typed sections
* **Compatibility**: New features in pure-Python syntax (annotations, support for `final`, `readonly`, etc.) may only work with recent Cython 3 releases

## Core Features and Syntax

The table below summarizes the differences between core features in both syntaxes. Both approaches assume the code includes `import cython`.

| Feature | Pure Python Syntax | Cython Syntax |
|---------|-------------------|---------------|
| File Extension | `.py` | `.pyx` |
| Static Typing | Variable annotations (`i: cython.int`) | `cdef` keyword (`cdef int i`) |
| C Functions | Decorators (`@cython.cfunc`, `@cython.ccall`) | `cdef`/`cpdef` keywords (`cdef int f()`) |
| Class Definition | Decorator (`@cython.cclass`) with Python `class` | `cdef class` keyword (`cdef class MyClass:`) |
| Struct, Union | `cython.struct`, `cython.union` | `cdef struct`, `cdef union` |
| Include Statement | Not Supported | `include "filename.pxi"` |
| Memory Management | Memoryviews | Memoryviews |
| Accessing C Variables/Pointers | `x.address` for pointers, `value[0]` to dereference | Direct using `&` operator |

## Example 1: Static Typing

### Pure Python

```python
import cython

def add(a: cython.int, b: cython.int) -> cython.int:
    return a + b
```

### Cython Syntax

```python
def add(int a, int b):
    return a + b
```

## Example 2: C Functions

### Pure Python

```python
import cython

@cython.cfunc
def c_add(a: cython.int, b: cython.int) -> cython.int:
    return a + b

@cython.ccall
def hybrid_add(a: cython.int, b: cython.int) -> cython.int:
    return a + b
```

### Cython Syntax

```python
cdef int c_add(int a, int b):
    return a + b

cpdef int hybrid_add(int a, int b):
    return a + b
```

## Example 3: Extension Types (C Classes)

### Pure Python

```python
import cython

@cython.cclass
class Rectangle:
    width: cython.int
    height: cython.int

    def __init__(self, width: cython.int, height: cython.int):
        self.width = width
        self.height = height

    def area(self) -> cython.int:
        return self.width * self.height
```

### Cython Syntax

```python
cdef class Rectangle:
    cdef int width
    cdef int height

    def __init__(self, int width, int height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
```

## Example 4: C Structures

### Pure Python

```python
import cython

Point = cython.struct(x=cython.int, y=cython.int)

def distance(p1: Point, p2: Point) -> cython.double:
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
```

### Cython Syntax

```python
cdef struct Point:
    int x
    int y

def distance(Point p1, Point p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
```

## Example 5: Exception Propagation

### Pure Python

```python
import cython

@cython.cfunc
@cython.exceptval(-1, check=True)
def pySum(a: cython.int, b: cython.int) -> cython.int:
    return a + b
```

### Cython Syntax

```python
cdef int pySum(int a, int b) except? -1:
    return a + b
```

## Example 6: Accessing C Variables and Pointers

### Pure Python

```python
import cython

cdef extern from *:
    void test_function(void* a)

cdef struct test_struct:
    int a
    int b

def main(test: test_struct):
    a = cython.address(test)
    b = cython.cast(cython.p_int, test.a)
    deref = b[0]
```

### Cython Syntax

```python
cdef extern from *:
    void test_function(void* a)

cdef extern struct test_struct:
    int a
    int b

def main(test_struct test):
    test_function(&test)  # Pass by Reference using Address-of operator
    deref = (&test.a)[0]
```

## Conclusion

Both pure Python and Cython syntax have their advantages:

- **Pure Python syntax** provides better compatibility with Python tooling and ecosystem (linters, IDEs, etc.)
- **Cython syntax** is more concise for heavily typed code and provides more explicit control for C/C++ integration

Your choice between these approaches should be based on your project requirements, team familiarity with Cython, need for Python compatibility, and the extent of C/C++ library integration needed.

For new projects using Cython 3+, the pure Python syntax offers excellent compatibility with standard Python tools while still providing most of Cython's performance benefits.
