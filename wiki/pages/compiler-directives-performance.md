# Compiler Directives: Performance

Directive combinations used in hot kernels.

## Typical Set

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True
```

## Notes

- Combine with typed loops and contiguous access patterns.
- For parallel code, ensure `nogil` regions stay pure C.
- Re-run correctness and benchmarks after toggling directives.

## See Also

- [[parallelism-prange]]
- [[memoryviews-contiguity-performance]]
- [[optimization-hot-loops]]
