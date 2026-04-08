# Extension Types

`cdef class` — C-level classes with typed attributes and fast method dispatch.

## Overview

Extension types are Cython's equivalent of C structs with methods. Attributes
are stored as C fields (no `__dict__`), giving direct memory access. Use them
for data structures (trees, heaps, buffers) and stateful computation.

## Syntax

```cython
cdef class Particle:
    cdef double x, y, vx, vy
    cdef readonly double mass        # readable from Python, not writable

    def __cinit__(self, double x, double y, double mass):
        self.x = x; self.y = y; self.mass = mass
        self.vx = 0.0; self.vy = 0.0

    def __dealloc__(self):
        pass  # free any malloc'd memory here

    cdef void step(self, double dt):  # C-only method (fast)
        self.x += self.vx * dt
        self.y += self.vy * dt

    cpdef double energy(self):        # callable from both C and Python
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)

    @property
    def position(self):
        return (self.x, self.y)

# Inheritance
cdef class ChargedParticle(Particle):
    cdef double charge

# Decorators
@cython.final               # prevents subclassing, enables faster dispatch
cdef class FastAccum:
    cdef double total

@cython.freelist(64)         # reuses deallocated objects (small, frequent allocs)
cdef class Point:
    cdef double x, y
```

## Patterns

### __cinit__ vs __init__

`__cinit__` is called before the object is fully constructed — use it for C-level initialization (malloc, field setup). `__init__` is the normal Python initializer.

```cython
cdef class Buffer:
    cdef int *data
    cdef int size

    def __cinit__(self, int size):
        # Called FIRST — guaranteed even if __init__ fails
        # Use for C allocation
        self.data = <int *>malloc(size * sizeof(int))
        if not self.data:
            raise MemoryError()
        self.size = size
        memset(self.data, 0, size * sizeof(int))

    def __init__(self, int size):
        # Called SECOND — for Python-level setup
        pass

    def __dealloc__(self):
        # Always called — even if __init__ failed
        if self.data:
            free(self.data)
```

**Key rule**: `__cinit__` always runs, `__dealloc__` always runs. Pair malloc in `__cinit__` with free in `__dealloc__` for leak-free RAII.

### Special Methods Must Use `def`

This is a common trace error (16+ occurrences across numerical categories):

```cython
# BAD — cdef for special methods
cdef class Node:
    cdef int value

    cdef int __len__(self):       # ERROR: Special methods must be declared with 'def', not 'cdef'
        return 1

    cdef str __repr__(self):      # ERROR: same issue
        return f"Node({self.value})"

# GOOD — special methods always use def
cdef class Node:
    cdef int value

    def __len__(self):
        return 1

    def __repr__(self):
        return f"Node({self.value})"
```

Special methods include: `__len__`, `__getitem__`, `__setitem__`, `__contains__`, `__iter__`, `__next__`, `__repr__`, `__str__`, `__hash__`, `__eq__`, `__lt__`, `__add__`, etc.

### Attribute Access Control

```cython
cdef class Config:
    cdef public int width, height     # read/write from Python
    cdef readonly str name            # read-only from Python
    cdef double _internal              # C-only (invisible from Python)

    def __cinit__(self, str name, int w, int h):
        self.name = name
        self.width = w
        self.height = h
        self._internal = 0.0
```

### RAII Pattern for Manual Memory

Pairing allocation with deallocation in cdef class:

```cython
from libc.stdlib cimport malloc, free

cdef class SparseMatrix:
    cdef double *values
    cdef int *col_idx
    cdef int *row_ptr
    cdef int nnz, rows

    def __cinit__(self, int rows, int nnz):
        self.rows = rows
        self.nnz = nnz
        self.values = <double *>malloc(nnz * sizeof(double))
        self.col_idx = <int *>malloc(nnz * sizeof(int))
        self.row_ptr = <int *>malloc((rows + 1) * sizeof(int))
        if not self.values or not self.col_idx or not self.row_ptr:
            # __dealloc__ will clean up whatever was allocated
            raise MemoryError()

    def __dealloc__(self):
        if self.values: free(self.values)
        if self.col_idx: free(self.col_idx)
        if self.row_ptr: free(self.row_ptr)
```

### @cython.final for Performance

Final classes can't be subclassed, enabling direct method calls (no vtable lookup):

```cython
@cython.final
cdef class IntStack:
    cdef int *data
    cdef int size, capacity

    def __cinit__(self, int capacity=16):
        self.data = <int *>malloc(capacity * sizeof(int))
        if not self.data:
            raise MemoryError()
        self.size = 0
        self.capacity = capacity

    cdef void push(self, int val) noexcept:   # can be inlined due to @final
        if self.size >= self.capacity:
            self.capacity *= 2
            self.data = <int *>realloc(self.data, self.capacity * sizeof(int))
        self.data[self.size] = val
        self.size += 1

    cdef int pop(self) noexcept:
        self.size -= 1
        return self.data[self.size]

    def __dealloc__(self):
        if self.data: free(self.data)
```

### @cython.freelist for High-Throughput Allocation

For small objects that are allocated and freed frequently:

```cython
@cython.freelist(256)
cdef class TreeNode:
    cdef int value
    cdef TreeNode left, right

    def __cinit__(self, int value):
        self.value = value
        self.left = None
        self.right = None
```

The freelist caches 256 deallocated TreeNode objects and reuses them, avoiding malloc/free overhead.

## Trace Statistics

Across ~2,800 traces:

| Error Pattern | Count | Categories |
|--------------|-------|------------|
| Special methods declared with cdef | 16+ | numerical, algorithms |
| C struct/union/enum definition not allowed here | 1+ | compression |

## Gotchas

1. **`def` for special methods** — `__len__`, `__getitem__`, `__repr__`, etc. MUST use `def`, not `cdef` or `cpdef`.
2. **__cinit__ vs __init__** — Use `__cinit__` for C-level init (malloc). It runs even if the subclass __init__ fails.
3. **__dealloc__ runs always** — Even if __cinit__ partially fails (some allocs succeeded, some didn't). Check for NULL before free.
4. **No __dict__ by default** — cdef class attributes must be declared. To allow arbitrary Python attrs, add `cdef dict __dict__`.
5. **Inheritance constraints** — cdef classes can only inherit from other cdef classes (or object). Cannot inherit from regular Python classes.
6. **cpdef methods incur overhead** — If a method is only called from Cython, use `cdef` for faster dispatch. Use `cpdef` only when Python needs to call it too.

## See Also

[[typing]], [[memory-management]], [[c-interop]], [[pitfalls]]
