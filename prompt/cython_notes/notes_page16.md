**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 16: Naming Conventions**

Consistent naming conventions are *very* important for readability and maintainability. Cython can sometimes blur Python and C worlds together, so being consistent makes it easy to quickly see what is what.

**General Principles:**

*   **Clarity:** Choose names that clearly convey the variable or function's purpose.
*   **Consistency:** Adhere to a single, well-defined naming scheme throughout your project.
*   **Readability:**  Names should be easy to pronounce and understand.
*  **Uniqueness Conventions**: Using consistent name prefixes based on data structures helps easily find when Cython syntax is used.

**Specific Conventions:**

*   **C-only Functions (`cdef` or `@cfunc`):** Prefix with an underscore (`_`). Example: `_c_my_function()`.

*   **C Struct Variables:** Suffix with `_s`. Example: `person_s`. Pure Python Annotation form is unnecessary.

*   **C Pointer Variables:** Suffix with `_p`.  Example: `data_p`, `matrix_p`. Annotation: `variable: cython.p_double`.
*   **Python Variables:** Follow normal Pythonic conventions (e.g., `my_variable`, `myFunction()`).

*   Return Python Objects **From pure C code**: All such objects must begin with `Py`. Allocation from them to C names requires allocation and NULL testing prior.

**When Sharing Declarations (``.pxd`` Files):**

*   **.pxd File Matching:** The name of functions and declarations in the ``.pxd`` should match the implementation/declaration *EXACTLY*
If discrepancies are required as shown in :ref:`resolve-conflicts` it must be implemented and accounted for.

**Example:**

```cython
# in my_module.pxd
cdef int _c_calculate_area(int width, int height)

# in my_module.pyx
cdef int _c_calculate_area(int width, int height):
    return width * height

def calculate_area(width, height):
    return _c_calculate_area(width, height)
```

**Rationale:**

*   **Underscores for C-only:** Signal that these functions are not directly callable from Python code (consistent with Python's convention for "private" members).
*   **`_s` suffix:**  Immediately identify C struct variables, making code easier to understand.
*   **`_p` suffix:** Quickly show it's a pointer, which has implications for memory management.
*  **Use the ``def`` function to wrap C code in Python**: Explicitly use the Python function to invoke all callable C names.

**Note:**  While these are just *conventions*, adhering to them consistently can greatly improve code clarity and reduce errors.
