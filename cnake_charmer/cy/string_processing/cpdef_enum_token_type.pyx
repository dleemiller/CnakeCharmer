# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Tokenize generated text by character type using cpdef enum (Cython-optimized).

Keywords: string processing, tokenizer, cpdef enum, character classification, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cpdef enum TokenType:
    ALPHA = 0
    DIGIT = 1
    SPACE = 2
    PUNCT = 3
    OTHER = 4


cdef int _is_punct(int code) noexcept:
    # . , ; : ! ? -
    if code == 46 or code == 44 or code == 59 or code == 58 or code == 33 or code == 63 or code == 45:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(100000,))
def cpdef_enum_token_type(int n):
    """Classify characters into token types using cpdef enum."""
    cdef int i, code
    cdef int alpha_count = 0
    cdef int digit_count = 0
    cdef int space_count = 0
    cdef int punct_count = 0
    cdef int other_count = 0

    for i in range(n):
        code = (i * 73 + 17) % 128
        if (65 <= code <= 90) or (97 <= code <= 122):
            alpha_count += 1
        elif 48 <= code <= 57:
            digit_count += 1
        elif code == 32:
            space_count += 1
        elif _is_punct(code):
            punct_count += 1
        else:
            other_count += 1

    return (alpha_count, digit_count, space_count, punct_count, other_count)
