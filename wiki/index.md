# Wiki Index

## Language Features
- [Typing: Declarations](pages/typing-declarations.md) — cdef/cpdef declarations and aliases
- [Typing: Common Errors](pages/typing-common-errors.md) — placement, redeclaration, cimport issues
- [Extension Types: Lifecycle](pages/extension-types-lifecycle.md) — __cinit__/__dealloc__ patterns
- [Extension Types: Performance](pages/extension-types-performance.md) — final/freelist and hot fields
- [Enums and C-Tuples](pages/enums-tuples.md) — cdef enum, cpdef enum, C-tuples
- [Error Handling: Except Clauses](pages/error-handling-except-clauses.md) — choosing except, except?, except *, noexcept
- [Error Handling: nogil and Callbacks](pages/error-handling-nogil-callbacks.md) — with gil, callback signature compatibility
- [Error Handling: C Cleanup](pages/error-handling-c-cleanup.md) — cleanup and failure paths with malloc/free

## Data Access
- [Memoryviews: Casting and Shapes](pages/memoryviews-casting-shapes.md) — pointer casts and dimensionality
- [Memoryviews: Contiguity and Performance](pages/memoryviews-contiguity-performance.md) — layout and loop order
- [NumPy Interop: Imports and Types](pages/numpy-interop-imports.md) — import + cimport requirements
- [NumPy Interop: dtypes and Allocation](pages/numpy-interop-dtype-allocation.md) — dtype matching and reuse
- [Memory Management: Allocation](pages/memory-management-allocation.md) — allocation and realloc safety patterns
- [Memory Management: Cleanup](pages/memory-management-cleanup.md) — try/finally and cascading cleanup
- [Memory Management: Buffer Protocol](pages/memory-management-buffer-protocol.md) — safe exported buffers

## Interop
- [C Interop: Imports and Paths](pages/c-interop-imports-paths.md) — libc paths and cimport rules
- [C Interop: Structs and extern](pages/c-interop-structs-extern.md) — struct/union and extern declarations
- [C Interop: Function Pointers](pages/c-interop-function-pointers.md) — callback signatures and noexcept
- [C++ Interop: STL Containers](pages/cpp-interop-stl.md) — vector/map/priority_queue patterns
- [C++ Interop: Classes and Exceptions](pages/cpp-interop-classes-exceptions.md) — wrappers and except+

## Performance
- [Parallelism: prange](pages/parallelism-prange.md) — loop parallelism and reduction behavior
- [Parallelism: nogil and Callbacks](pages/parallelism-nogil-callbacks.md) — helper signatures and callback safety
- [Compiler Directives: Core Safety](pages/compiler-directives-core.md) — correctness-impacting directives
- [Compiler Directives: Performance](pages/compiler-directives-performance.md) — hot-loop directive sets
- [Optimization: Annotation Score](pages/optimization-annotation-score.md) — targeting fallback lines
- [Optimization: Hot Loops](pages/optimization-hot-loops.md) — loop-level optimization priorities

## Reference
- [Pitfalls: Compile-Time Errors](pages/pitfalls-compile-errors.md) — recurring compile failures
- [Pitfalls: Runtime and Safety](pages/pitfalls-runtime-safety.md) — leaks, NULL checks, pointer safety
- [Pitfalls: Parallel and nogil](pages/pitfalls-parallel-nogil.md) — prange/nogil pitfalls
