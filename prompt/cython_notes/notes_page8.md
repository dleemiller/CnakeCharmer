
**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 8: NumPy Integration**

NumPy and Cython work exceptionally well together. NumPy's powerful array operations combined with Cython's speed enhancements produce incredibly fast number-crunching code.

*   **Why NumPy Integration?** It's rare to find high-performance numeric Python code *not* leveraging NumPy.

**1. Basic NumPy Array Usage (Legacy Buffer Protocol):**

*   **Cython Syntax:** (Avoid) `cnp.ndarray[DTYPE_t, ndim=2]` (Use the MemoryView version from Chapter 4, instead)
*   **`cimport numpy`**: Include to gain access to numpy's C-level API.
*   **`import_array()`**:  Call this *after* `cimport numpy` to initialize NumPy's C API. Failing to call this is source to many issues.

**2. Optimizing Array Access (Memoryviews are Preferred):**

*   **Reason:** Standard NumPy indexing uses Python objects, which adds overhead. Memoryviews directly access the underlying data.
*   **`@cython.boundscheck(False)`**: *Carefully* disable bounds checking for loops over NumPy arrays in situations from page 4 Memoryviews.
*   **`@cython.wraparound(False)`**: *Carefully* disable negative index wrapping.

**3. Key Optimization Points:**

*   **Profiling:**  Always profile to locate bottlenecks before optimizing.
*   **Type everything**: Declare C types for indices, loop counters, and temporary variables.
*   **Data Contiguity:** If guaranteed, declare C or Fortran contiguity in NumPy arrays using memoryviews and the `::1` syntax. For example: `cdef double[:, ::1] array` (C-contiguous). `cdef double[::1, :] array` (Fortran-contiguous).

**4. Memory Management Considerations:**

*   When allocating memory from C for passing to NumPy, utilize a memoryview so that deallocation is handled during the runtime.
*   NumPy has special compilation types such as `np.intc`, `np.double`

**5. NumPy Universal Functions (ufuncs)**

* **Purpose** ufuncs take special functions to apply element-wise to one or more arrays.
* **Cython Decorator** `@cython.ufunc`. Tag a cdef/cfunc to apply a compiled function in an efficient loop.
* **Arguments** Scalar Python Objects or numeric values.
* **Fused Types** Use fused types for ufunc functions with multiple return values.
* **Multithreading** Declare `nogil` to have the threads release the GIL, when the Numpy headers available upon C compilation upon first generation.

**6. Limitations of Static NumPy Typing :**

* Only supports fast element indexing of single elements.
* Does not speed up whole-array operations
* Does not speed up calls to NumPy global functions or methods.
