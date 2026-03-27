# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply linear transforms using cdef class with __call__ (Cython).

Keywords: numerical, callable, transform, linear, cdef class, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class LinearTransform:
    """Callable linear transform: result = scale * x + offset."""
    cdef double scale
    cdef double offset

    def __cinit__(self, double scale, double offset):
        self.scale = scale
        self.offset = offset

    def __call__(self, double x):
        return self.scale * x + self.offset

    cdef inline double apply(self, double x):
        return self.scale * x + self.offset


@cython_benchmark(syntax="cy", args=(200000,))
def callable_transform(int n):
    """Create transform objects and apply to n values, summing results."""
    cdef list transforms = []
    cdef double scale, offset, val, total
    cdef int i
    cdef LinearTransform t

    for i in range(16):
        scale = ((<long long>i * <long long>2654435761 + 17) % 1000) / 100.0 - 5.0
        offset = ((<long long>i * <long long>1103515245 + 12345) % 1000) / 50.0 - 10.0
        transforms.append(LinearTransform(scale, offset))

    total = 0.0
    for i in range(n):
        val = ((<long long>i * <long long>1664525 + <long long>1013904223) % 1000000) / 1000.0
        t = <LinearTransform>transforms[i & 15]
        total += t.apply(val)

    return total
