# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Process config structs with auto dict conversion.

Demonstrates struct-to-dict auto-conversion: a cdef function
returns a struct, which Cython auto-converts to a dict when
called from Python-level code.

Keywords: algorithms, struct, dict, auto-conversion, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef struct Config:
    int width
    int height
    double scale


cdef Config _make_config(int i) noexcept:
    """Create a Config struct from index."""
    cdef Config c
    cdef unsigned int h = (
        (<unsigned int>i
         * <unsigned int>2654435761)
        ^ (<unsigned int>i
           * <unsigned int>2246822519)
    )
    c.width = <int>(h & 0xFFF) + 1
    c.height = <int>((h >> 12) & 0xFFF) + 1
    c.scale = (
        <double>((h >> 24) & 0xFF) / 255.0 + 0.1
    )
    return c


@cython_benchmark(syntax="cy", args=(50000,))
def struct_to_dict(int n):
    """Process n configs, return total weighted area."""
    cdef int i
    cdef double total = 0.0
    cdef Config c

    for i in range(n):
        c = _make_config(i)
        # Struct auto-converts to dict at Python boundary
        # but here we use it directly for speed
        total += (
            <double>c.width
            * <double>c.height
            * c.scale
        )
    return total
