# C++ Interop

Using C++ STL containers, classes, and templates from Cython.

## Overview

Cython can wrap C++ classes, use STL containers (vector, map, set), handle
C++ exceptions, and instantiate templates. Requires `# distutils: language = c++`
or auto-detection via `from libcpp` imports.

## Syntax

```cython
# STL containers
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.set cimport set as cppset
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.queue cimport priority_queue

cdef vector[int] v
v.push_back(42)

cdef cppmap[int, double] m
m[1] = 3.14

# Wrapping a C++ class
cdef extern from "myclass.h":
    cdef cppclass MinHeap:
        MinHeap() except +
        void push(int val) except +
        int pop() except +
        bint empty()

# Inline C++ class
cdef extern from *:
    """
    struct Vec2D {
        double x, y;
        Vec2D operator+(const Vec2D& o) const { return {x+o.x, y+o.y}; }
    };
    """
    cdef cppclass Vec2D:
        double x, y
        Vec2D operator+(Vec2D)

# Exception handling
cdef extern from "lib.h":
    double divide(double a, double b) except +              # translates any C++ exception
    double safe_div(double a, double b) except +ValueError  # maps to specific Python exception
```

## Patterns

### STL Containers as Fast Internal Data Structures

STL containers can replace Python dicts/lists in hot code without GIL overhead:

```cython
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

def count_elements(int[::1] data):
    cdef unordered_map[int, int] counts
    cdef int i, n = data.shape[0]

    for i in range(n):
        counts[data[i]] += 1    # no Python dict overhead

    # Convert back to Python only at the end
    return dict(counts)
```

### priority_queue for Graph Algorithms

Common in graph/algorithms traces for Dijkstra, A*, etc.:

```cython
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

# BAD — using priority_queue without cimport
cdef priority_queue[int] pq     # ERROR: 'priority_queue' is not a type identifier

# GOOD — proper cimport
from libcpp.queue cimport priority_queue

# Min-heap pattern (negate values since STL priority_queue is max-heap)
cdef priority_queue[pair[double, int]] pq
pq.push(pair[double, int](-dist, node))

while not pq.empty():
    top = pq.top()
    pq.pop()
    cost = -top.first
    node = top.second
```

### Nullary Constructor Requirement

C++ classes must have a default (nullary) constructor to be stack-allocated in Cython:

```cython
# BAD — no default constructor
cdef extern from *:
    """
    struct Foo {
        Foo(int x) : val(x) {}  // no default constructor!
        int val;
    };
    """
    cdef cppclass Foo:
        Foo(int x) except +

cdef Foo f  # ERROR: C++ class must have a nullary constructor to be stack allocated

# GOOD — add a default constructor
cdef extern from *:
    """
    struct Foo {
        Foo() : val(0) {}
        Foo(int x) : val(x) {}
        int val;
    };
    """
    cdef cppclass Foo:
        Foo() except +
        Foo(int x) except +

cdef Foo f       # stack allocated with default constructor
cdef Foo f2 = Foo(42)  # or with argument
```

This error appeared in compression traces.

### C++ Exception Translation

The `except +` clause translates C++ exceptions to Python exceptions:

```cython
cdef extern from "lib.h":
    double divide(double a, double b) except +              # any C++ exception → RuntimeError
    int lookup(int key) except +IndexError                  # map to specific Python exception
    void process() except +*                                # custom handler

# If the C++ function throws std::bad_alloc → MemoryError
# If it throws std::exception → RuntimeError with what() message
# If it throws anything else → RuntimeError("Unknown exception")
```

### When C++ Interop Helps vs Pure C

**Use C++** when:
- You need hash maps, priority queues, or sorted containers (STL is well-optimized)
- You're wrapping an existing C++ library
- You need RAII-style resource management (destructors)
- Graph algorithms that need dynamic data structures

**Use pure C** when:
- Maximum portability needed
- Simple array-based algorithms
- nogil blocks where you don't need containers
- You want to avoid C++ compilation overhead

### Converting STL Containers to Python

```cython
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap

# vector → Python list (automatic conversion)
cdef vector[int] v
v.push_back(1)
v.push_back(2)
py_list = list(v)    # [1, 2]

# map → Python dict (automatic conversion)
cdef cppmap[int, double] m
m[1] = 3.14
py_dict = dict(m)    # {1: 3.14}
```

## Gotchas

1. **Type identifier not found** — `priority_queue`, `vector`, etc. need `from libcpp.X cimport X`. Without cimport, you get `'priority_queue' is not a type identifier`.
2. **Nullary constructor** — Stack-allocated C++ objects need a default constructor. Use `new`/`del` for heap allocation if no default constructor exists.
3. **except + required** — C++ constructors and methods that might throw need `except +` or exceptions are silently swallowed.
4. **Language directive** — C++ features require `# distutils: language = c++` in the file or build configuration.
5. **STL in nogil** — Most STL operations are C++ and work without GIL, but iterating with Python conversion requires GIL.

## See Also

[[c-interop]], [[typing]], [[memory-management]], [[error-handling]]
