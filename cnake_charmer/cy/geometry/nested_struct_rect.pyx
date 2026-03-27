# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Rectangle intersection using nested structs.

Demonstrates nested struct: Point inside Rect.
Computes intersection area of n rectangle pairs.

Keywords: geometry, rectangle, intersection, nested struct, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef struct Point:
    double x
    double y


cdef struct Rect:
    Point min_pt
    Point max_pt


cdef inline double _max(double a, double b) noexcept:
    if a > b:
        return a
    return b


cdef inline double _min(double a, double b) noexcept:
    if a < b:
        return a
    return b


@cython_benchmark(syntax="cy", args=(50000,))
def nested_struct_rect(int n):
    """Compute total intersection area of n rect pairs."""
    cdef int i
    cdef unsigned int h1, h2, h3, h4
    cdef Rect a, b
    cdef double ix0, iy0, ix1, iy1
    cdef double total_area = 0.0

    for i in range(n):
        h1 = <unsigned int>i * <unsigned int>2654435761
        h2 = <unsigned int>i * <unsigned int>2246822519
        h3 = (
            <unsigned int>(i + 1)
            * <unsigned int>2654435761
        )
        h4 = (
            <unsigned int>(i + 1)
            * <unsigned int>2246822519
        )

        a.min_pt.x = (
            <double>(h1 & 0xFFFF) / 65535.0
            * 100.0
        )
        a.min_pt.y = (
            <double>((h1 >> 16) & 0xFFFF)
            / 65535.0 * 100.0
        )
        a.max_pt.x = (
            a.min_pt.x
            + <double>(h2 & 0xFFFF)
            / 65535.0 * 20.0 + 1.0
        )
        a.max_pt.y = (
            a.min_pt.y
            + <double>((h2 >> 16) & 0xFFFF)
            / 65535.0 * 20.0 + 1.0
        )

        b.min_pt.x = (
            <double>(h3 & 0xFFFF) / 65535.0
            * 100.0
        )
        b.min_pt.y = (
            <double>((h3 >> 16) & 0xFFFF)
            / 65535.0 * 100.0
        )
        b.max_pt.x = (
            b.min_pt.x
            + <double>(h4 & 0xFFFF)
            / 65535.0 * 20.0 + 1.0
        )
        b.max_pt.y = (
            b.min_pt.y
            + <double>((h4 >> 16) & 0xFFFF)
            / 65535.0 * 20.0 + 1.0
        )

        ix0 = _max(a.min_pt.x, b.min_pt.x)
        iy0 = _max(a.min_pt.y, b.min_pt.y)
        ix1 = _min(a.max_pt.x, b.max_pt.x)
        iy1 = _min(a.max_pt.y, b.max_pt.y)

        if ix0 < ix1 and iy0 < iy1:
            total_area += (ix1 - ix0) * (iy1 - iy0)

    return total_area
