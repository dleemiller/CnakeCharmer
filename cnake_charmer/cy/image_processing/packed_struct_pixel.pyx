# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Process RGBA pixels with packed struct.

Uses cdef packed struct Pixel with no padding between
unsigned char fields. Applies gamma and alpha blending.

Keywords: image processing, pixel, packed struct, RGBA, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef packed struct Pixel:
    unsigned char r
    unsigned char g
    unsigned char b
    unsigned char a


@cython_benchmark(syntax="cy", args=(100000,))
def packed_struct_pixel(int n):
    """Process n RGBA pixels, return checksum."""
    cdef int i
    cdef unsigned int h, checksum
    cdef unsigned int rv, gv, bv

    cdef Pixel *pixels = <Pixel *>malloc(
        n * sizeof(Pixel)
    )
    if not pixels:
        raise MemoryError()

    for i in range(n):
        h = (
            <unsigned int>i
            * <unsigned int>2654435761
        )
        pixels[i].r = <unsigned char>(h & 0xFF)
        pixels[i].g = <unsigned char>(
            (h >> 8) & 0xFF
        )
        pixels[i].b = <unsigned char>(
            (h >> 16) & 0xFF
        )
        pixels[i].a = <unsigned char>(
            (h >> 24) & 0xFF
        )

    checksum = 0
    for i in range(n):
        rv = <unsigned int>pixels[i].r
        gv = <unsigned int>pixels[i].g
        bv = <unsigned int>pixels[i].b

        # Gamma: val * val / 255
        rv = (rv * rv) / 255
        gv = (gv * gv) / 255
        bv = (bv * bv) / 255

        # Alpha premultiply
        rv = (rv * <unsigned int>pixels[i].a) / 255
        gv = (gv * <unsigned int>pixels[i].a) / 255
        bv = (bv * <unsigned int>pixels[i].a) / 255

        checksum += (
            rv + gv + bv
            + <unsigned int>pixels[i].a
        )

    free(pixels)
    return <long long>(checksum & <unsigned int>0xFFFFFFFF)
