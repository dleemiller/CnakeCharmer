# Pitfalls: Compile-Time Errors

Frequent compile-time failures and quick corrections.

## Top Errors

- `cdef statement not allowed here`
- `'np' is not a cimported module`
- `'libc/stdlib/memset.pxd' not found`
- `Exception clause not allowed for function returning Python object`

## Patterns

### cdef placement

```cython
# BAD
x = 0
cdef int i

# GOOD
cdef int i
x = 0
```

### import vs cimport

```cython
# BAD
import numpy as np
cdef np.npy_intp i

# GOOD
import numpy as np
cimport numpy as np
cdef np.npy_intp i
```

### wrong libc path

```cython
# BAD
# from libc.stdlib cimport memset

# GOOD
from libc.string cimport memset
```

## See Also

- [[typing-declarations]]
- [[c-interop-imports-paths]]
- [[compiler-directives-core]]
