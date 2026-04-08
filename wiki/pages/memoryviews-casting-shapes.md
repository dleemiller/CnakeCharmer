# Memoryviews: Casting and Shapes

Converting pointers safely and matching dimensionality.

## Patterns

### pointer to memoryview

```cython
cdef double *ptr = ...
cdef double[:] view = <double[:n]> ptr
```

### avoid void* direct conversion

Cast `void*` to a typed pointer first, then to a memoryview slice.

### shape compatibility

Do not assign 1D buffers to 2D typed memoryviews without reshape/reinterpretation.

## See Also

- [[memory-management-allocation]]
- [[numpy-interop-dtype-allocation]]
