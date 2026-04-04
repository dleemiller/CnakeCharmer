# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the Number Theoretic Transform (NTT) (Cython-optimized).

Keywords: ntt, number theoretic transform, fft, modular arithmetic, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

DEF NTT_MOD = 998244353
DEF NTT_G = 3


@cython_benchmark(syntax="cy", args=(10000,))
def number_theoretic_transform(int n):
    """Compute NTT of sequence a[i] = (i*7+3) % 998244353, return sum mod p.

    Uses C arrays for in-place iterative Cooley-Tukey NTT.

    Args:
        n: Length of the input sequence.

    Returns:
        Sum of all transformed values, mod 998244353.
    """
    cdef int size, i, j, bit, length, half, k
    cdef long long w, wn, u, v, total
    cdef long long base, exp_val, result

    # Round up to next power of 2
    size = 1
    while size < n:
        size <<= 1

    cdef long long *arr = <long long *>malloc(size * sizeof(long long))
    if arr == NULL:
        raise MemoryError()

    # Build input
    for i in range(n):
        arr[i] = (i * 7 + 3) % NTT_MOD
    for i in range(n, size):
        arr[i] = 0

    # Bit-reversal permutation
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            u = arr[i]
            arr[i] = arr[j]
            arr[j] = u

    # Iterative NTT
    length = 2
    while length <= size:
        half = length >> 1
        # w = pow(NTT_G, (NTT_MOD - 1) // length, NTT_MOD)
        base = NTT_G
        exp_val = (NTT_MOD - 1) // length
        result = 1
        while exp_val > 0:
            if exp_val & 1:
                result = result * base % NTT_MOD
            base = base * base % NTT_MOD
            exp_val >>= 1
        w = result

        for i in range(0, size, length):
            wn = 1
            for k in range(half):
                u = arr[i + k]
                v = arr[i + k + half] * wn % NTT_MOD
                arr[i + k] = (u + v) % NTT_MOD
                arr[i + k + half] = (u - v + NTT_MOD) % NTT_MOD
                wn = wn * w % NTT_MOD
        length <<= 1

    total = 0
    for i in range(size):
        total = (total + arr[i]) % NTT_MOD

    free(arr)
    return int(total)
