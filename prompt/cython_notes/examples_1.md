``Canonical Examples: Python vs. Cython``
===========================================

``Overview``
-----------

This page provides a set of canonical examples comparing implementations in pure Python versus Cython. The examples showcase different language features and usage scenarios, with a focus on performance improvements gained with Cython's static typing and direct C code generation.
The Cython examples will have 2 code segments, a pure Python form and the cython-ized code.
The intention is to showcase the differences in performance and readability when types are added.

.. note::
    These examples are aimed at demonstrating the core concepts of Cython. Actual
    performance gains will vary based on the specific use-case, hardware, and
    compiler optimizations applied during the build process. Profile before optimizing.

``Example 1: Basic Arithmetic``
------------------------------

**Aim**: Demonstrate performance differences in basic arithmetic operations.

**Python**

.. code-block:: python

   def py_add(a, b):
       return a + b

**Cython**

.. code-block:: cython
    def add(int a, int b):
        return a + b

**Performance Comparison**: Adding static types to a simple addition provides a small but measurable speedup. Because the interpreter overhead has been removed.

``Example 2: Looping with Range``
-------------------------------

**Aim**: Showcase the impact of static typing on loop performance.

**Python**

.. code-block:: python

   def py_sum_range(n):
       s = 0
       for i in range(n):
           s += i
       return s

**Cython**

.. code-block:: cython

   def sum_range(int n):
        cdef int i, s = 0
        for i in range(n):
            s += i
        return s
**Performance Comparison**: With both index and result variables defined as a C type we can improve the loop speed significantly, where larger loops will show a greater speedup.

``Example 3: Array Processing with NumPy``
----------------------------------------

**Aim**: Demonstrate efficient array manipulation using memoryviews.

  .. code-block:: python

    import numpy as np
    def py_array_sum(arr):
        sum_val = 0
    n = arr.shape[0]
        for i in range(n):
            sum_val += arr[i]
        return sum_val

**Cython**

 .. code-block:: cython
    import numpy as np
    cimport numpy as np
    import cython
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def array_sum(np.ndarray[np.int_t, ndim=1] arr):
        cdef np.int_t sum = 0
        cdef int i
assert arr.dtype == np.intc

        for i in range(arr.shape[0]):
          sum = sum + arr[ i ]
        return sum

**Performance comparison**: Accessing numpy arrays with memory views gives a significant boost for numerical operations. As in previous examples, turning off bounds checking further increases performance where it is safe to do so.

``Example 4: Struct Usage``
--------------------------

**Aim**: Illustrate the performance benefits of using C structs for data grouping.

**Python**: (This will be an artificial case given the lack of struct support...)

.. code-block:: python
    def py_process_points(points):  #Imagine points is a list of (x, y) tuples
        result = 0
        for p in points:
          x, y = p
        result += point_function(x, y)
        return result

**Cython**:
*First, define the C struct in Cython:*

.. code-block:: cython

    cdef struct Point:
        int x
        int y

Then, define a function using the struct:

.. code-block:: cython

   cdef int point_function(Point p):
       return p.x * p.y

    def process_points(list points):
        cdef int result = 0
        cdef Point p ##We now know that the "point" is of an extension type rather than a Python object.

        for point in points:
        result += point_function(p)
        return result

**Performance Comparison**: Using C structures provides benefits in terms of memory layout and direct access to data, enhancing the performance for calculations on complex data structures. The downside is a loss of flexibility compared to dynamically typed objects.

``Example 5: Callbacks``
----------------------

**Aim**: Illustrate the use of callback functions in C Libraries

*Declare the function called in C Header:*

.. code-block:: c

    typedef int (*callback_t)(int);
    int call_with(int x, call_back_t foo);

**Python**:
*Since the C Function takes a function pointer as input, annotate with "noexcept"*

.. code-block:: cython

    cdef extern int call_with(int x, object foo) noexcept

Then you can call it:

      .. code-block:: cython
            def callCFunc(bar):
                cdef int y = call_with(5, bar)
                return y

**Performance Comparison**: Use the @exceptval(check=False) decorator for faster function creation.

``Conclusion``
---------------

These simple examples provide a starting point to understand the potential advantages that the combination of Python and Cython can offer. Cython's ability to seamlessly integrate Python and C allows for significant performance optimizations while maintaining a high level of code readability and flexibility. By strategically adding static types and leveraging Cython's features, developers get the best of both worlds.
