# Parallelism: nogil and Callbacks

`nogil` helper and callback signature rules.

## Rules

1. Helpers called from `nogil` should be `noexcept nogil` when they cannot raise.
2. C callbacks must match pointer signature exactly.
3. Avoid implicit exception checks in callback paths.

## Pattern

```cython
cdef int cmp(const void *a, const void *b) noexcept nogil:
    ...
```

## See Also

- [[c-interop-function-pointers]]
- [[error-handling-nogil-callbacks]]
