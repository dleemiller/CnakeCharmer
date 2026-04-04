# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""PNG filter undo operations: sub, up, average, paeth (Cython-optimized).

Uses C arrays with malloc/free for scanline processing.

Keywords: png, filter, sub, up, average, paeth, compression, cython, benchmark
"""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef inline int c_abs(int x) noexcept nogil:
    if x < 0:
        return -x
    return x


cdef inline unsigned char paeth_predictor(int a, int b, int c) noexcept nogil:
    cdef int p, pa, pb, pc
    p = a + b - c
    pa = c_abs(p - a)
    pb = c_abs(p - b)
    pc = c_abs(p - c)
    if pa <= pb and pa <= pc:
        return <unsigned char>a
    elif pb <= pc:
        return <unsigned char>b
    else:
        return <unsigned char>c


cdef unsigned char * generate_scanlines(int width, int height, int filter_unit) noexcept nogil:
    """Generate deterministic scanline data as flat C array."""
    cdef int row_bytes = width * filter_unit
    cdef int total = height * row_bytes
    cdef unsigned char *data = <unsigned char *>malloc(total)
    cdef int y, x, idx
    for y in range(height):
        for x in range(row_bytes):
            idx = y * row_bytes + x
            data[idx] = <unsigned char>(((y * 131 + x * 37 + 7) * 73) % 256)
    return data


cdef unsigned char * filter_sub(unsigned char *src, int height, int row_bytes, int filter_unit) noexcept nogil:
    cdef int total = height * row_bytes
    cdef unsigned char *out = <unsigned char *>malloc(total)
    cdef int y, x, idx
    cdef unsigned char filt, left
    for y in range(height):
        for x in range(row_bytes):
            idx = y * row_bytes + x
            filt = src[idx]
            if x >= filter_unit:
                left = out[idx - filter_unit]
            else:
                left = 0
            out[idx] = <unsigned char>((filt + left) & 0xFF)
    return out


cdef unsigned char * filter_up(unsigned char *src, int height, int row_bytes) noexcept nogil:
    cdef int total = height * row_bytes
    cdef unsigned char *out = <unsigned char *>malloc(total)
    cdef int y, x, idx
    cdef unsigned char filt, above
    for y in range(height):
        for x in range(row_bytes):
            idx = y * row_bytes + x
            filt = src[idx]
            if y > 0:
                above = out[(y - 1) * row_bytes + x]
            else:
                above = 0
            out[idx] = <unsigned char>((filt + above) & 0xFF)
    return out


cdef unsigned char * filter_average(unsigned char *src, int height, int row_bytes, int filter_unit) noexcept nogil:
    cdef int total = height * row_bytes
    cdef unsigned char *out = <unsigned char *>malloc(total)
    cdef int y, x, idx
    cdef unsigned char filt
    cdef int left, above
    for y in range(height):
        for x in range(row_bytes):
            idx = y * row_bytes + x
            filt = src[idx]
            if x >= filter_unit:
                left = out[idx - filter_unit]
            else:
                left = 0
            if y > 0:
                above = out[(y - 1) * row_bytes + x]
            else:
                above = 0
            out[idx] = <unsigned char>((filt + ((left + above) >> 1)) & 0xFF)
    return out


cdef unsigned char * filter_paeth(unsigned char *src, int height, int row_bytes, int filter_unit) noexcept nogil:
    cdef int total = height * row_bytes
    cdef unsigned char *out = <unsigned char *>malloc(total)
    cdef int y, x, idx
    cdef unsigned char filt
    cdef int left, above, upper_left
    for y in range(height):
        for x in range(row_bytes):
            idx = y * row_bytes + x
            filt = src[idx]
            if x >= filter_unit:
                left = out[idx - filter_unit]
            else:
                left = 0
            if y > 0:
                above = out[(y - 1) * row_bytes + x]
            else:
                above = 0
            if x >= filter_unit and y > 0:
                upper_left = out[(y - 1) * row_bytes + x - filter_unit]
            else:
                upper_left = 0
            out[idx] = <unsigned char>((filt + paeth_predictor(left, above, upper_left)) & 0xFF)
    return out


cdef long checksum(unsigned char *data, int size) noexcept nogil:
    cdef long total = 0
    cdef int i
    for i in range(size):
        total += data[i]
    return total


@cython_benchmark(syntax="cy", args=(200, 200, 3))
def png_filters(int width, int height, int filter_unit):
    """Apply all 4 PNG reconstruction filters and return checksums."""
    cdef int row_bytes = width * filter_unit
    cdef int total = height * row_bytes
    cdef unsigned char *scanlines
    cdef unsigned char *res_sub
    cdef unsigned char *res_up
    cdef unsigned char *res_avg
    cdef unsigned char *res_paeth
    cdef long cs_sub, cs_up, cs_avg, cs_paeth

    with nogil:
        scanlines = generate_scanlines(width, height, filter_unit)
        res_sub = filter_sub(scanlines, height, row_bytes, filter_unit)
        res_up = filter_up(scanlines, height, row_bytes)
        res_avg = filter_average(scanlines, height, row_bytes, filter_unit)
        res_paeth = filter_paeth(scanlines, height, row_bytes, filter_unit)

        cs_sub = checksum(res_sub, total)
        cs_up = checksum(res_up, total)
        cs_avg = checksum(res_avg, total)
        cs_paeth = checksum(res_paeth, total)

        free(scanlines)
        free(res_sub)
        free(res_up)
        free(res_avg)
        free(res_paeth)

    return (cs_sub, cs_up, cs_avg, cs_paeth)
