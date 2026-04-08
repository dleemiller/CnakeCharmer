# NumPy Interop: Imports and Types

Import patterns required for typed NumPy usage.

## Pattern

```cython
import numpy as np
cimport numpy as np

cdef np.ndarray[np.float64_t, ndim=1] arr
cdef np.npy_intp i
```

## Gotchas

- `import` alone is not enough for typed Cython declarations.
- Missing `cimport` triggers `'np' is not a cimported module`.

## See Also

- [[typing-declarations]]
- [[memoryviews-casting-shapes]]
