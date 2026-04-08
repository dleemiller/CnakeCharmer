# Pitfalls: Parallel and nogil

Common `prange`/`nogil` mistakes.

## Patterns

### Python object usage in nogil

```cython
# BAD: tuple/list/dict operations in nogil
# GOOD: use C scalars, structs, pointers, memoryviews
```

### Invalid `prange` arguments

`reduction=` is not a valid keyword in Cython `prange`; reductions are inferred from `+=`, `*=`, etc.

### Callback mismatch

C callbacks with `except *` signatures fail type checks; use `noexcept`.

## See Also

- [[parallelism-prange]]
- [[parallelism-nogil-callbacks]]
- [[c-interop-function-pointers]]
