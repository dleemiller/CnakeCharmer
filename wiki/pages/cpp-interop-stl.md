# C++ Interop: STL Containers

Working with `vector`, `map`, `priority_queue`, and related types.

## Pattern

```cython
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue

cdef vector[int] v
v.push_back(1)
```

## Gotchas

- Missing `cimport` causes unknown type errors.
- `priority_queue` is max-heap by default.
- Container-to-Python conversion can add overhead; do it at boundaries.

## See Also

- [[cpp-interop-classes-exceptions]]
- [[optimization-hot-loops]]
