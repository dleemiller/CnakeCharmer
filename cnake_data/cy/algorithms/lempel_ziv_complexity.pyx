# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Lempel-Ziv complexity measure for binary sequences.

Keywords: lempel ziv, kolmogorov complexity, sequence complexity, compression, cython
"""

from libc.math cimport log
from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef double _lz_complexity(const char *s, int n) nogil:
    """Compute normalized Lempel-Ziv complexity of a char array."""
    if n == 0:
        return 0.0

    cdef int c = 1
    cdef int l = 1
    cdef int i = 0
    cdef int k = 1
    cdef int k_max = 1
    cdef double b

    while True:
        if i + k - 1 >= n or l + k - 1 >= n:
            break
        if s[i + k - 1] != s[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l >= n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k - 1 >= n:
                c += 1
                break

    b = n / log(<double>n) * log(2.0) if n > 1 else 1.0
    return c / b


@cython_benchmark(syntax="cy", args=(1000,))
def lempel_ziv_complexity(int n):
    """Compute LZ complexity for n deterministic binary sequences.

    Args:
        n: Number of sequences to analyze.

    Returns:
        Tuple of (total_complexity, max_complexity, min_complexity).
    """
    cdef double total = 0.0
    cdef double max_c = 0.0
    cdef double min_c = 1e300
    cdef double c_val
    cdef int i, j, bit
    cdef int seq_len = 200

    cdef char *seq = <char *>malloc(seq_len * sizeof(char))
    if not seq:
        raise MemoryError()

    for i in range(n):
        for j in range(seq_len):
            bit = ((i * 7 + j * 13 + i * j * 3 + 5) % 97) & 1
            seq[j] = <char>(48 + bit)

        c_val = _lz_complexity(seq, seq_len)
        total += c_val
        if c_val > max_c:
            max_c = c_val
        if c_val < min_c:
            min_c = c_val

    free(seq)
    return (total, max_c, min_c)
