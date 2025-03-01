**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 3: Function Types and Optimization**

Cython offers three levels of function definition, each balancing accessibility with speed:

*   **`def` functions:** These are standard Python functions. They accept and return Python objects, ensuring they are fully callable from Python code. Static typing of parameters increases performance.

*   **`cdef` functions:** These are C-only functions. They are the fastest function type, as they operate directly with C types and avoid Python object overhead. They *cannot* be called directly from Python code.

*   **`cpdef` functions:**  These offer a hybrid approach. They create two entry points: a fast C entry point for calls from Cython code and a Python wrapper for calls from Python. They combine the benefits of both accessibility and performance, with a slight overhead related to virtual function tables compared to the `cdef` versions.

**How to pick:**
    *   If pure python compatibility is needed, use `def`
    *   If the function is called only in Cython, use `cdef`
    *   If you need both, use `cpdef`

**Return Type Declarations:**

*   Providing a return type, if that is a primitive other than Python object, allows for greater optimization, compared to python `object`.

**Optimization Strategies:**

*   **Profiling:** Identify where performance is lacking. The profiler will show where slow areas are.

*   **Static Typing:** Add C types to variables and function signatures, which can dramatically increase speed, especially within loops. Type at minimum index variables of quick arrays are best, parameters and local variables related to the operations performed in that loop.

*   **Inline Functions:** Use `cdef inline` for small, frequently called functions to eliminate function call overhead.

*   **Cython Annotations:** Use the `-a` flag on `cython` to generate an HTML report displaying lines that translate to C vs. lines that call internal Cython functions. This helps to identify areas suitable for optimization.  White lines typically offer the greatest potential for optimization through static typing.
