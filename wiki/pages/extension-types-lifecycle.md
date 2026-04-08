# Extension Types: Lifecycle

Construction, initialization, and teardown patterns.

## Pattern

```cython
cdef class Buffer:
    cdef double *data
    cdef Py_ssize_t n

    def __cinit__(self, Py_ssize_t n):
        self.n = n
        self.data = <double *>malloc(n * sizeof(double))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            free(self.data)
```

## Gotchas

- If `__cinit__` fails mid-way, cleanup must still be safe.
- Never assume Python-level attributes exist in `__cinit__`.
- Keep ownership simple: one class owns one allocation.

## See Also

- [[memory-management-allocation]]
- [[memory-management-cleanup]]
