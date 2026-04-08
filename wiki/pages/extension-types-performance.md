# Extension Types: Performance

Patterns for low-overhead extension types.

## Rules

1. Prefer typed cdef fields for frequently accessed values.
2. Keep hot methods in `cdef`/`cpdef` where callable context allows.
3. Use `@cython.final` when inheritance is not needed.
4. Use `@cython.freelist(N)` for high-churn object allocation patterns.

## Gotchas

- Special methods still must be `def`.
- Excess Python-object fields reduce gains from extension types.

## See Also

- [[typing]]
- [[optimization-hot-loops]]
