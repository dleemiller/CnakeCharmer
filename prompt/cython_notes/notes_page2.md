**Cython Cliffs Notes: Essential Code Topics for Python Programmers**

**Page 2: Pure Python vs Cython Syntax**

Cython offers two main syntax options for marking variables and functions with C information: pure Python syntax and Cython syntax. Pure Python syntax is newer and more versatile, though Cython syntax is more common and generally shorter. 

**Pure Python Syntax**

*   **Requires:** This newer style of coding uses Python's built-in type hint capabilities (PEP-484 like annotations). It is a Python3 way of working and you must include `cimport cython` at the beginning of your program.
*   **Benefits:** Is compatible with all Python syntax, thus making your code fully functional in standard Python3 interpreter. 
*   **Code Style Example: Function**

```python
import cython

@cython.cfunc
def add_one(x: cython.int) -> cython.int:
    return x + 1
```

*   **Code Style Example: Variable**

```python
x: cython.long = 1
s: cython.p_char
```

**Cython Syntax**

*   **Requires:** This older style has custom Cython keywords for integrating C/C++ types into your Python code. The program files typically have a '.pyx' or '.pxd' extension.
*   **Benefits:** More familiar to C/C++ developers, and easier for C/C++ functions. The extension types used are less strict.
*   **Code Style Example: Function**

```cython
cdef inline int minimum(int a, int b):
    return a if a <= b else b
```

*   **Code Style Example: Variable**

```cython
cdef int age = 30
cdef float gpa
cdef double * grades
```

**Considerations**

*   **When to use Pure Python:** If you ever want to run code with a standard Python interpreter, you must use pure Python syntax. Code can also be more understandable among Python programmers since it’s just standard annotated code.
*   **When to use Cython:** If you need a quick and dirty prototype with lots of support functions, Cython syntax can be good for adding quick type definitions and function declarations from C/C++. 
*   **When to use a mixture of both syntaxes:** If you prefer annotation style type definitions but also want to take full advantages of external C/C++ APIs, you'll often find yourself intermixing type declarations and C imports from Pure Python files.
