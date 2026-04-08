# Memoryviews: Contiguity and Performance

Using contiguous layouts for predictable speed.

## Rules

1. Use C-contiguous views for row-major traversal.
2. Align loop order with memory layout.
3. Disable bounds/wrap checks only after correctness is confirmed.

## Pattern

```cython
# cython: boundscheck=False, wraparound=False
cdef double[:, ::1] a  # C-contiguous second axis
```

## See Also

- [[compiler-directives-performance]]
- [[optimization-hot-loops]]
