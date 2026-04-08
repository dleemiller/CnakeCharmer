# Memory Management: Buffer Protocol

Expose C-allocated memory to Python consumers safely.

## Overview

When a `cdef class` exports a buffer, ownership and lifetime must be explicit.

## Pattern

```cython
cdef class DoubleBuffer:
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

    def __getbuffer__(self, Py_buffer *buf, int flags):
        buf.buf = <void *>self.data
        buf.len = self.n * sizeof(double)
        buf.itemsize = sizeof(double)
        buf.ndim = 1

    def __releasebuffer__(self, Py_buffer *buf):
        pass
```

## Gotchas

- Do not free memory while exported buffers may still exist.
- Ensure shape/strides/format metadata is correct.
- Keep ownership in one class to prevent lifetime confusion.

## See Also

- [[memoryviews]]
- [[numpy-interop]]
- [[memory-management-cleanup]]
