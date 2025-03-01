**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 5: Extension Types (cdef classes)**

Extension types, also known as `cdef classes`, are a special kind of class in Cython that closely resemble C structs. They bring significant memory and performance advantages over normal Python classes because they store data directly in memory, therefore circumventing python overheads.

**Key Characteristics:**

*   **C Struct Based:** Store data directly in a C struct instead of a Python dictionary.
*   **Static Attributes:** Attributes *must* be pre-declared with C types using `cdef`, annotations or `cython.declare`, can be `public` (read+write from Python), `readonly` (read-only from Python), or private(c access only).
*   **cdef/cpdef Methods:**  Support both C-level (`cdef`) and Python-accessible (`cpdef`) methods.

**Benefits:**

*   Significant performance increase due to direct memory access and C-level operations.
*   Efficient memory usage.
*   Easy interfacing with C code.

**Differences from Python Classes:**

*   Cannot dynamically add attributes unless `__dict__` is included.
*   Require explicit allocation and deallocation of C resources via `__cinit__` (constructor), `__dealloc__` (destructor). Note, `__init__` and `__del__` are still supported but work as standard Python code.
* Base class must also be a cdef class or a C built in type.

**Key Elements:**

*   **`cdef class`:** Declares an extension type. Pure Python Annotation form is `@cython.cclass`
*   **`cdef`**: Declare C data type attributes (fields) of the class. This is what stores data directly in memory. Note: Python annotations style is valid here.
*   **`__cinit__`**: Special constructor for C attributes and initialisation.
*   **`__dealloc__`**: Special destructor for manual memory management with C.
Note that methods are declared as standard Python functions `def`.
*   **`cpdef`**: Declare function that combines characteristics from both Python and C.

**Example:**

```cython
cdef class Rectangle:
    cdef int width, height

    def __cinit__(self, int width, int height):
        self.width = width
        self.height = height

    cpdef int area(self):
        return self.width * self.height
```

**When to Use?**

*   When you need the highest possible performance and memory efficiency.
*   When interfacing directly with C libraries and data structures.
*   When your data model is fixed at compile time.

**Trade-offs:**

*   Less dynamic than Python classes.
*   Requires more manual memory management.
*   Potentially steeper learning curve.
