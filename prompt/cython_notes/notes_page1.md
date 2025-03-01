**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 1: Static Type Declarations**

Cython is a Python compiler that generates optimized C code, enabling significant speedups compared to standard Python. A key optimization technique involves using static type declarations. This allows Cython to bypass Python's dynamic nature and generate faster C code.

**1. Declaring Variables:**

*   **Benefits:** Static typing is not a necessity, instead optimize code for when needed. With static typing, Cython knows the type of variable at compile time, not run time, making code faster. 
*   **Speed:** Use typing when you need to perform arithmetic operations or access efficient indexing from memory views.
*   **C Syntax** Using the C syntax for static type declarations is more concise and readable from a C/C++ perspective. It is good for use in function arguments, parameters and return values.
*   **Pure Python Syntax** Using Pure Python syntax offers type hints that follow PEP 484 and PEP 526. Here you first import Cython, then you can call Cython datatypes and functions to specify the types for local variables and attributes in classes.
*   **cdef Keyword:** In Cython syntax, the main keyword is `cdef`, for variables and functions
*   **Inferred Typing:** Cython can infer types for variables assignments in a few scenarios.

**2. Data Types:**

*   **C Types:** Utilize standard C data types like `int`, `float`, `double`, `char`, along with their `unsigned` variants. They must be exact to produce correct C code.
*   **Python Types:** You can type Python Objects themselves such as `list`, `dict` and user created classes, with limitations to account for performance.
*   **ctuples:** Create efficient alternatives to Python tuples by compiling a tuple with valid C types.

**3. Functions**

*   **Python (`def`):** Use normal Python functions. Pass Python objects as parameters and return values. Can be called from Python.
*   **C (`cdef` or `@cfunc`):**  Specify type if returning non-Python object. Python functions are not accessible from function, will return Python objects which can execute when execution leaves without a return value or `0` for standard C values with other return types.
*   **Hybrid (`cpdef` or `@ccall`):** Combines both `def` and `cdef` characteristics, which can be called from both Cython and Python with their respective characteristics.

**4. Identifying Bottlenecks:**

*   **Profiling:** Identify where the code is spending the most time running
*   **Cython Annotations:** Use the -a flag on `cython` compiler to produce an HTML report that highlights which lines will translate to C and which will translate to Python. This will help identify where it's possible and/or necessary to add type information.
