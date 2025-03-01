**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 17: Function Design Patterns**

Effective function design is critical for both performance and maintainability in Cython. These patterns help balance C efficiency with Python usability:

*   **Small, Single-Purpose `cdef` Functions:**

    *   Keep `cdef` functions concise and focused on a specific task.
    *   Use them as building blocks within larger algorithms.
    *   If a function has multiple use cases, consider creating multiple smaller `cdef` functions instead of one overly complex function.
    *   Makes inlining more effective as inlined instructions should be related.

*   **Python-Facing Wrappers:**

    *   Create Python-facing wrappers for `cdef` function using standard `def` (for backwards compatible methods) or `cpdef` (for new methods which might be shared through inheritance). These are used to wrap pure C code logic in Python context.
    *   For example: functions that are python compatible can have a base return type of Python `object`

*   **Error Handling:**

    *   In C-only (`cdef`) functions, return explicit error codes or sentinel values, since you can't throw Python exceptions. C functions can also return C enums or other Python types
    *   Python compatibility requires error handlers to call standard exception constructors and type verifications in def/cpdef

*   **When to Use hybrid (`cpdef` or `@ccall`)**

    *   Use them when you're looking to maximize code reuse and maintainability in a larger design.
    * Benefits: 
        1. Can be accessed from both languages 
        2. Retains C speed when function is called from Cython
        3. The same function design can exist anywhere in code.
*    **Pure Python Mode:** If Python3 and Cython3 are available, use the pure Python mode as a convenient replacement.

*   **Typing of positional and Keyword arguments**
    *   In `def` functions allow typed, positional argument to be as general as possible.
    *   If access speed is important, declare as Cython compatible datatypes, but it can complicate some things.
    *  If a Hybrid type is needed, declare both `*args` and `**kwargs` to wrap argument of a `cdef` for Python API to Python methods and C code that relies on Cython's functions.
