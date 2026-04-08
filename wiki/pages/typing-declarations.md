# Typing: Declarations

Declaration rules for variables, functions, and aliases.

## Patterns

```cython
cdef int i
cdef double total = 0.0

cdef inline double dot(double* a, double* b, int n) noexcept nogil:
    ...
```

- Keep declarations before executable statements.
- Use fused types only where generic performance is needed.
- Prefer simple concrete types in hot kernels.

## See Also

- [[typing-common-errors]]
- [[optimization-annotation-score]]
