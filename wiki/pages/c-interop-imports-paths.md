# C Interop: Imports and Paths

Correctly importing C declarations in Cython.

## Overview

Most C interop compile failures come from wrong import style or wrong libc path.

## Rules

1. `cimport` for C-level declarations.
2. `import` for Python-level modules.
3. Use libc paths by header, not by function.

## Patterns

### NumPy typed declarations require cimport

```cython
import numpy as np
cimport numpy as np

cdef np.npy_intp i
```

### Common libc paths

```cython
from libc.stdlib cimport malloc, free, calloc, realloc, qsort
from libc.string cimport memcpy, memset, memcmp, strlen
from libc.math cimport sin, cos, sqrt, log, exp, M_PI
from libc.stdint cimport uint8_t, uint32_t, uint64_t
```

### Avoid wrong granularity

```cython
# BAD
# from libc.stdlib cimport memset

# GOOD
from libc.string cimport memset
```

## Reference Table

- `malloc`, `free`, `calloc`, `realloc`: `libc.stdlib`
- `memcpy`, `memset`, `memcmp`, `strlen`: `libc.string`
- `sin`, `cos`, `sqrt`, `log`, `exp`: `libc.math`
- fixed-width ints: `libc.stdint`

## See Also

- [[c-interop-structs-extern]]
- [[memory-management-allocation]]
- [[pitfalls]]
