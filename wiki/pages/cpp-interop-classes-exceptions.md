# C++ Interop: Classes and Exceptions

Class wrapping and C++ exception translation.

## Pattern

```cython
cdef extern from "lib.h" namespace "ns":
    cdef cppclass Foo:
        Foo() except +
        int step(int x) except +
```

## Rules

1. Keep method signatures aligned with headers.
2. Use `except +` when C++ exceptions should become Python exceptions.
3. Add explicit default constructors when wrappers require them.

## See Also

- [[error-handling-except-clauses]]
- [[cpp-interop-stl]]
