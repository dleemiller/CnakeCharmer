``Syntax Comparison: Pure Python vs. Cython``
==============================================

``Overview``
-----------

Cython provides two main syntax variations for writing its code: pure Python syntax (using type hints and decorators) and Cython syntax (using `cdef` and other keywords). This page highlights the primary differences between the two approaches.

The key considerations when choosing a syntax are:
* **Readability**: Pure Python syntax allows the .py file to be valid Python code.
* **Flexibility**: Pure Python allows easy refactoring in Python without specific Cython knowledge.
* **C/C++ Integration**: Cython syntax provides better and more explicit control.
* **Ease of Use**: Pure Python might be easier for beginners, while Cython syntax is generally considered more concise in typed sections;
* **Backwards Compatibility**: New features in pure-python syntax such as annotations, support for final, readonly, and more may work only for recent Cython 3 releases.

``Core Features and Syntax``
-----------------------------

The table below show the differences between core features and syntax. It builds on the assumption that both files contains the line  `import cython`.

+-------------------------+--------------------------------------------+-----------------------------------------------------+
| Feature                 | Pure Python Syntax                         | Cython Syntax                                       |
+=========================+============================================+=====================================================+
| File Extension          | ``.py``                                    | ``.pyx``                                             |
+-------------------------+--------------------------------------------+-----------------------------------------------------+
| Static Typing           | Variable annotations and decorators (e.g.,  | ``cdef`` keyword (e.g., ``cdef int i``)               |
|                         | ``i: cython.int``)                        |                                                     |
+-------------------------+--------------------------------------------+-----------------------------------------------------+
| C Functions             | Decorator (``@cython.cfunc`` /  ``ccall``) | ``cdef`` / ``cpdef`` keyword (e.g.,  ``cdef int f()``) |
+-------------------------+--------------------------------------------+-----------------------------------------------------+
| Class Definition        | Decorator (``@cython.cclass``) with        | ``cdef class`` (e.g., ``cdef class MyClass:``)       |
|                         | Python ``class``                           |                                                     |
+-------------------------+--------------------------------------------+-----------------------------------------------------+
| Struct, Union           | ``cython.struct``,  ``cython.union``          | ``cdef struct``,  ``cdef union``                       |
+-------------------------+--------------------------------------------+-----------------------------------------------------+
| Include Statment       |Not Supported. | ``include "filename.pxi"``   |                                                     |
+--------------------------------+--------------------------------------------+-----------------------------------------------------+
| Memory management  | Memoryviews (see tutorial)           | Memoryviews (see tutorial)    |
+-------------------------+--------------------------------------------+-----------------------------------------------------+|
| Accessing C-variables and pointers| Direct access with  ``x.address`` function to access pointer or ``value[0]`` to dereference|  Direct  using & operator, Indirect access is not supported|

``Example 1: Static Typing``
----------------------------

**Pure Python**

.. code-block:: python
    import cython
    def add(a: cython.int, b: cython.int) -> cython.int:
        return a + b

**Cython Syntax**

.. code-block:: cython
    def add(int a, int b):
        return a + b

``Example 2: C Functions``
--------------------------

**Pure Python**

.. code-block:: python
    import cython
    @cython.cfunc
    def c_add(a: cython.int, b: cython.int) -> cython.int:
        return a + b

    @cython.ccall
    def hybrid_add(a: cython.int, b: cython.int) -> cython.int:
        return a + b

**Cython Syntax**

.. code-block:: cython
    cdef int c_add(int a, int b):
        return a + b

    cpdef int hybrid_add(int a, int b):
        return a + b

``Example 3: Extension Types (C Classes)``
-----------------------------------------

**Pure Python**

.. code-block:: python
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

**Cython Syntax**

.. code-block:: cython
    cdef class Rectangle:
        cdef int width
        cdef int height

        def __init__(self, int width, int height):
            self.width = width
            self.height = height

        def area(self):
            return self.width * self.height

``Example 4: C Structures``
--------------------------

**Pure Python**

.. code-block:: python
    import cython

    Point = cython.struct(x=cython.int, y=cython.int)

    def distance(p1: Point, p2: Point) -> cython.double:
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

**Cython Syntax**

.. code-block:: cython
    cdef struct Point:
        int x
        int y

    def distance(Point p1, Point p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

``Example 5: Exception Propagation``
-----------------------------

**Pure Python**

.. code-block:: python
        import cython

        @cython.cfunc
        @cython.exceptval(-1, check=True)
        def pySum(a: cython.int, b: cython.int) -> cython.int:
                return a + b

**Cython Syntax**

.. code-block:: cython
        cdef int pySum(int a, int b) except? -1:
                return a + b

``Example 6: Direct accessing C variables and pointers``
-----------------------------

**Pure Python**

.. code-block:: python
    import cython
    cdef extern from *:
        void test_function(void* a)
    cdef struct test_struct:
        int a
        int b
    def main(test:test_struct):
        a = cython.address(test)
        b = cython.cast(cython.p_int, test.a)
        deref = b[0]
   

**Cython Syntax**

.. code-block:: cython
     cdef extern from *:   
        void test_function(void* a)
    cdef extern struct test_struct:
        int a
        int b

    def main(test_struct test):   
        test_function(&test)  //Pass by Reference using Address-of operator   
        deref = (&test.a)[ 0 ]     

``Conclusion``
---------------

Both pure Python and Cython syntax have advantages. Pure Python syntax provides greater compatibility with Python tooling and ecosystem, while Cython syntax is more verbose but can be more expressive for C/C++ integration. Choosing between these two approaches depends on factors such as project needs, team skills, readability, and C/C++ library integration needs.
