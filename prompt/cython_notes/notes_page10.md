**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 10: Parallelization with `prange`**

Cython provides several built-in mechanisms for leveraging multi-core processing. The easiest of these is the `prange` function, which simplifies parallelizing loops.  This requires the GIL to released using the `nogil` option. Cython will then parallelize the outmost loop.

**Key Features:**

*   **Syntax:** Similar to Python's `range`, but designed for parallel execution. 
*   **Requires:**
    *   Release GIL, use with the `nogil` clause in a `with` statement, or function argument.
    *   The bodies of parallel loops must be *GIL-free*. Cannot directly manipulate Python objects. Use C types.
    * Make sure to enable OpenMP during compilation by passing to the C or C++ compiler flags such as `-fopenmp` (for GCC) or `/openmp` (for MSVC).

*   **Accessibility:** Available through `cimport cython.parallel`.

**Basic Usage:**

```python
from cython.parallel import prange

def my_func(int n):
    cdef int i, sum = 0
    #Using nogil since only C variables are used
    for i in prange(n, nogil=True): 
        sum += i 
    return sum
```

*   **Parameters (Simplified):**
    *   `start`:  Loop start value.
    *   `stop`:  Loop stop value (exclusive).
    *   `step`:  Loop increment.
    *   `nogil=True`: Crucial!  Releases the GIL for parallel processing.
    *  `use_threads_id= CONDITION`: If you assign to a variable in a `prange` block, it becomes ``lastprivate``, meaning that the variable will contain value of last sequential.

*   **Loop Variables:** Loop variables (e.g., `i` in the example) *must* be explicitly typed as C integers.

**Reduction:**

*   `prange` automatically handles reduction operations (e.g., `+=`, `*=`, `-=`) on C scalar variables. The final result gets accumulated.
*   This is also a must for `parrange` when passing thread-local buffers.

**Optional: Scheduling & Chunking:**

*   `schedule`: Specifies the OpenMP scheduling algorithm. common options: `static`, `dynamic`, `guided`, `runtime`, `auto`
*   `chunksize`: The chunk size used for each thread’s portion of loop iterations.
*   `num_threads`: Sets the number of threads the OpenMP should use.

**``Parallel``**

*   `Thread-Local buffers` Can invoke multiple blocks of code at once with the ``parallel`` directive with ``prange`` for sequential loops.
*   `OpenMP API Functions` Also by invoking C import ``openmp``.
