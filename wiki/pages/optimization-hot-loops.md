# Optimization: Hot Loops

Practical loop-level optimization priorities.

## Priority Order

1. Algorithmic complexity first.
2. C types and contiguous data access.
3. `nogil`/`prange` where safe and profitable.
4. C math/library functions over Python calls in loops.

## Patterns

- Precompute constants outside loops.
- Avoid per-iteration allocation.
- Keep branch structure simple in tight loops.

## See Also

- [[parallelism-prange]]
- [[compiler-directives-performance]]
- [[numpy-interop-dtype-allocation]]
