# NumPy Interop: dtypes and Allocation

Avoid dtype mismatch and allocation overhead.

## Rules

1. Match array dtype to typed view (`float64`, `int32`, etc.).
2. Allocate once and reuse in iterative loops.
3. Use memoryviews for element-wise loops; use NumPy vectorization where appropriate.

## Pattern

```cython
cdef double[:] view = arr  # arr must be float64-compatible
```

## See Also

- [[memoryviews]]
- [[optimization-hot-loops]]
