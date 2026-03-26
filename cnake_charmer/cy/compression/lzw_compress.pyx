# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LZW compression dictionary size computation (Cython-optimized).

Keywords: compression, lzw, dictionary, encoding, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def lzw_compress(int n):
    """Count dictionary entries created during LZW compression."""
    cdef int i, next_code
    cdef dict dictionary = {}
    cdef str w, c, wc

    # Initialize dictionary with single characters
    for i in range(4):
        dictionary[chr(65 + i)] = i

    next_code = 4
    w = chr(65 + (0 * 7 + 3) % 4)

    for i in range(1, n):
        c = chr(65 + (i * 7 + 3) % 4)
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            dictionary[wc] = next_code
            next_code += 1
            w = c

    return len(dictionary)
