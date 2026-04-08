# Optimization: Annotation Score

Using Cython annotations to target Python overhead.

## Rules

1. Focus first on yellow lines in hot loops.
2. Type loop indices and temporaries.
3. Replace Python container operations with C data paths where possible.
4. Verify improvements with `evaluate_cython` after each change.

## Pattern

```cython
cdef Py_ssize_t i
cdef double s = 0.0
for i in range(n):
    s += arr[i]
```

## Typical Gains

- Variable typing often yields the largest first jump.
- Switching from Python lists to memoryviews/pointers can remove persistent fallback lines.

## See Also

- [[typing-declarations]]
- [[memoryviews-casting-shapes]]
