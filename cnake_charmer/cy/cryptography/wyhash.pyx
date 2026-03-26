# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
wyhash fast hash function by Wang Yi (Cython-optimized).

Keywords: cryptography, hash, wyhash, mixing, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef unsigned long long _wyp0 = 0xa0761d6478bd642fULL
cdef unsigned long long _wyp1 = 0xe7037ed1a0b428dbULL
cdef unsigned long long _wyp2 = 0x8ebc6af09c88c6e3ULL
cdef unsigned long long _wyp3 = 0x589965cc75374cc3ULL


cdef unsigned long long _wymix(unsigned long long a, unsigned long long b):
    """128-bit multiply then xor-fold to 64 bits."""
    cdef unsigned long long lo, hi
    cdef object full = (<object>a) * (<object>b)
    lo = <unsigned long long>(full & 0xFFFFFFFFFFFFFFFF)
    hi = <unsigned long long>((full >> 64) & 0xFFFFFFFFFFFFFFFF)
    return lo ^ hi


cdef unsigned long long _wyread4(const unsigned char* data, int offset):
    """Read 4 bytes little-endian."""
    return (
        <unsigned long long>data[offset]
        | (<unsigned long long>data[offset + 1] << 8)
        | (<unsigned long long>data[offset + 2] << 16)
        | (<unsigned long long>data[offset + 3] << 24)
    )


cdef unsigned long long _wyread8(const unsigned char* data, int offset):
    """Read 8 bytes little-endian."""
    return (
        <unsigned long long>data[offset]
        | (<unsigned long long>data[offset + 1] << 8)
        | (<unsigned long long>data[offset + 2] << 16)
        | (<unsigned long long>data[offset + 3] << 24)
        | (<unsigned long long>data[offset + 4] << 32)
        | (<unsigned long long>data[offset + 5] << 40)
        | (<unsigned long long>data[offset + 6] << 48)
        | (<unsigned long long>data[offset + 7] << 56)
    )


cdef unsigned long long _wyhash_impl(const unsigned char* data, int length, unsigned long long seed):
    """Hash a byte buffer using wyhash."""
    cdef unsigned long long a, b, see1, see2
    cdef int idx = 0

    seed = seed ^ _wyp0

    if length <= 16:
        if length >= 4:
            a = _wyread4(data, 0) | (_wyread4(data, (length >> 3) << 2) << 32)
            b = _wyread4(data, length - 4) | (_wyread4(data, length - 4 - ((length >> 3) << 2)) << 32)
        elif length > 0:
            a = <unsigned long long>data[0] | (<unsigned long long>data[length >> 1] << 8) | (<unsigned long long>data[length - 1] << 16) | (<unsigned long long>length << 24)
            b = 0
        else:
            a = 0
            b = 0
        return _wymix(a ^ _wyp0, b ^ _wyp1) ^ (seed ^ <unsigned long long>length)

    see1 = seed
    if length > 48:
        see2 = seed
        while idx + 48 <= length:
            seed = _wymix(_wyread8(data, idx) ^ _wyp0, _wyread8(data, idx + 8) ^ seed)
            see1 = _wymix(_wyread8(data, idx + 16) ^ _wyp1, _wyread8(data, idx + 24) ^ see1)
            see2 = _wymix(_wyread8(data, idx + 32) ^ _wyp2, _wyread8(data, idx + 40) ^ see2)
            idx += 48
        seed = seed ^ see1 ^ see2

    while idx + 16 <= length:
        seed = _wymix(_wyread8(data, idx) ^ _wyp0, _wyread8(data, idx + 8) ^ seed)
        idx += 16

    a = _wyread8(data, length - 16)
    b = _wyread8(data, length - 8)
    return _wymix(a ^ _wyp1, b ^ seed) ^ <unsigned long long>length


@cython_benchmark(syntax="cy", args=(200000,))
def wyhash(int n):
    """Hash n different byte sequences using wyhash and return discriminating results.

    Args:
        n: Number of byte sequences to hash.

    Returns:
        Tuple of (total_hash_xor, last_hash, count_with_high_bit_set).
    """
    cdef unsigned long long total_xor = 0
    cdef unsigned long long last_hash = 0
    cdef unsigned long long high_bit_count = 0
    cdef unsigned long long h
    cdef int i, j, length
    cdef unsigned char buf[32]

    for i in range(n):
        length = (i % 31) + 1
        for j in range(length):
            buf[j] = (i * 7 + j * 13 + 5) & 0xFF
        h = _wyhash_impl(buf, length, 42ULL)
        total_xor = total_xor ^ h
        last_hash = h
        if h & (1ULL << 63):
            high_bit_count += 1

    return (total_xor, last_hash, high_bit_count)
