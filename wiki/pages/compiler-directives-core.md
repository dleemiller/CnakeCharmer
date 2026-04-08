# Compiler Directives: Core Safety

Directive behavior that affects correctness.

## Important

- `cdivision=True` changes division semantics and suppresses Python zero-division checks.
- `boundscheck=False` and `wraparound=False` remove safety checks; use after correctness validation.

## Pattern

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
```

## Gotchas

- malformed directive lines can produce confusing parse errors.
- unknown directives are rejected.

## See Also

- [[compiler-directives-performance]]
- [[pitfalls-compile-errors]]
