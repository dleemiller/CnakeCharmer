# Pitfalls: Runtime and Safety

Memory and pointer errors that compile but fail at runtime.

## Patterns

### Missing NULL checks

```cython
cdef int *buf = <int *>malloc(n * sizeof(int))
if not buf:
    raise MemoryError()
```

### Leaks on early return

```cython
try:
    # work
    return out
finally:
    free(buf)
```

### Unsafe pointer from temporary object

Keep a stable Python reference before taking C-level views/pointers.

## Fast Rules

1. Initialize pointers to `NULL`.
2. Use `try/finally` for owned allocations.
3. Do not cast temporary Python objects to primitive C pointers.

## See Also

- [[memory-management-allocation]]
- [[memory-management-cleanup]]
- [[error-handling-c-cleanup]]
