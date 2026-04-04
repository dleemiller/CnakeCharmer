"""Compute the Number Theoretic Transform (NTT) of a sequence.

Keywords: ntt, number theoretic transform, fft, modular arithmetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark

NTT_MOD = 998244353
NTT_G = 3  # primitive root of NTT_MOD


@python_benchmark(args=(10000,))
def number_theoretic_transform(n: int) -> int:
    """Compute NTT of sequence a[i] = (i*7+3) % 998244353, return sum mod 998244353.

    Pads the sequence length to the next power of 2, then applies the
    iterative Cooley-Tukey NTT in-place.

    Args:
        n: Length of the input sequence (will be padded to next power of 2).

    Returns:
        Sum of all transformed values, mod 998244353.
    """
    mod = NTT_MOD
    g = NTT_G

    # Round up to next power of 2
    size = 1
    while size < n:
        size <<= 1

    # Build input array
    arr = [(i * 7 + 3) % mod for i in range(n)]
    arr.extend([0] * (size - n))

    # Bit-reversal permutation
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    # Iterative NTT (Cooley-Tukey)
    length = 2
    while length <= size:
        half = length >> 1
        w = pow(g, (mod - 1) // length, mod)
        for i in range(0, size, length):
            wn = 1
            for k in range(half):
                u = arr[i + k]
                v = arr[i + k + half] * wn % mod
                arr[i + k] = (u + v) % mod
                arr[i + k + half] = (u - v) % mod
                wn = wn * w % mod
        length <<= 1

    total = 0
    for val in arr:
        total = (total + val) % mod

    return total
