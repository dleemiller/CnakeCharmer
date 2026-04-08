# Typing: Common Errors

Frequent typing-related compile failures.

## Top Errors

- `cdef statement not allowed here`
- variable redeclaration in same scope
- special methods declared as `cdef`
- missing `cimport` for typed symbols

## Quick Fixes

- Move all `cdef` declarations to top of function.
- Declare each variable once, then reuse.
- Use `def __len__(self)` and other special methods as `def`.
- Add required `cimport` lines for C/NumPy types.

## See Also

- [[pitfalls-compile-errors]]
- [[extension-types-lifecycle]]
