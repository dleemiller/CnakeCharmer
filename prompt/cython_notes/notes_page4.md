**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 4: Memoryviews**

Memoryviews allow efficient access to memory buffers (e.g., NumPy arrays) without Python overhead. They are a more general and feature-rich alternative to the older NumPy buffer support.

*   **Purpose:**  Provide optimized access to array-like data.

*   **Benefits:**
    *   Clean syntax, similar to NumPy slicing.
    *   Often operate without the GIL allowing multithreading.
    *   Faster than the old buffer syntax.
    *   Work with various buffer providers like NumPy, C arrays, Cython arrays, and Python's `array.array`.
    *   Compatible with both pure and standard Cython syntax

**Syntax & Usage:**

*   Declare a memoryview, specifying the data type and number of dimensions.
    *   *Example (Cython syntax):* `cdef double[:, :] matrix`
    *   *Example (Pure Python syntax):* `matrix: cython.double[:, :]`
*   Assign an object that exports the buffer interface to the memoryview.
    *   *Example:* `matrix = np.zeros((10, 10))`

**Accessing Elements:**

*   Use standard Python slicing syntax.

*   Index access is automatically translated into memory addresses (very fast).
    *   *Example:* `matrix[i, j] = 5.0`

*   Negative indices count from the end of the dimensions similar to Numpy.

**Contiguity:**

*   Specify memory layout for optimization of data access.
*   `::1` specifies contiguity in a dimension.
    *   `double[:, ::1]` - C contiguous (last dimension contiguous).
    *   `double[::1, :]` - Fortran contiguous (first dimension contiguous).
*   Assigning a non-contiguous buffer will raise a `ValueError` at runtime.
* `cython.view` includes memory layout contants, such as `generic`, `strided`, `indirect` and more

**Slicing:**

*   Memoryviews can be sliced in a similar way to NumPy arrays, creating new memoryviews. These do not own the buffer, similar to NumPy views.
    *   *Example:* `cdef double[:] view = matrix[0, :]`  *(Creates a 1D view of the first row)*
    *  Can use `None` keyword to insert arrays like in NumPy

*   Copy or `copy_fortran` methods to make data into C or Fortran contiguous slices respectively.
