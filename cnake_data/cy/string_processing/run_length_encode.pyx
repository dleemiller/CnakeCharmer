# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encode/decode a deterministic string.

Keywords: string processing, run-length encoding, compression, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def run_length_encode(int n):
    """Run-length encode a deterministic string and verify decode."""
    if n == 0:
        return (0, 0, 1)

    cdef char *chars = <char *>malloc(n * sizeof(char))
    if not chars:
        raise MemoryError()

    cdef int i, j
    cdef int num_runs = 1
    cdef int encoded_length = 0
    cdef char run_char
    cdef int run_len = 1
    cdef int digits, temp
    cdef int decoded_ok = 1
    cdef int decode_pos = 0

    # Build character array
    for i in range(n):
        chars[i] = 65 + (i * 3) % 5

    # Encode pass
    run_char = chars[0]
    for i in range(1, n):
        if chars[i] == run_char:
            run_len += 1
        else:
            digits = 0
            temp = run_len
            while temp > 0:
                digits += 1
                temp = temp / 10
            encoded_length += 1 + digits
            num_runs += 1
            run_char = chars[i]
            run_len = 1

    # Emit last run
    digits = 0
    temp = run_len
    while temp > 0:
        digits += 1
        temp = temp / 10
    encoded_length += 1 + digits

    # Decode verification pass
    run_char = chars[0]
    run_len = 1
    decode_pos = 0

    for i in range(1, n):
        if chars[i] == run_char:
            run_len += 1
        else:
            for j in range(run_len):
                if decode_pos + j >= n or chars[decode_pos + j] != run_char:
                    decoded_ok = 0
            decode_pos += run_len
            run_char = chars[i]
            run_len = 1

    for j in range(run_len):
        if decode_pos + j >= n or chars[decode_pos + j] != run_char:
            decoded_ok = 0
    decode_pos += run_len

    if decode_pos != n:
        decoded_ok = 0

    free(chars)
    return (encoded_length, num_runs, decoded_ok)
