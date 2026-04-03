"""Lempel-Ziv complexity measure for binary sequences.

Keywords: lempel ziv, kolmogorov complexity, sequence complexity, compression
"""

import math

from cnake_charmer.benchmarks import python_benchmark


def _lz_complexity(s):
    """Compute normalized Lempel-Ziv complexity of a string."""
    n = len(s)
    if n == 0:
        return 0.0

    c = 1
    pos = 1
    i = 0
    k = 1
    k_max = 1

    while True:
        if i + k - 1 >= n or pos + k - 1 >= n:
            break
        if s[i + k - 1] != s[pos + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == pos:
                c += 1
                pos += k_max
                if pos >= n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        else:
            k += 1
            if pos + k - 1 >= n:
                c += 1
                break

    b = n / math.log(n, 2) if n > 1 else 1.0
    return c / b


@python_benchmark(args=(1000,))
def lempel_ziv_complexity(n):
    """Compute LZ complexity for n deterministic binary sequences.

    Args:
        n: Number of sequences to analyze.

    Returns:
        Tuple of (total_complexity, max_complexity, min_complexity).
    """
    total = 0.0
    max_c = 0.0
    min_c = 1e300

    for i in range(n):
        # Generate deterministic binary sequence of length 200
        seq_len = 200
        chars = []
        for j in range(seq_len):
            bit = ((i * 7 + j * 13 + i * j * 3 + 5) % 97) & 1
            chars.append(chr(48 + bit))
        seq = "".join(chars)

        c = _lz_complexity(seq)
        total += c
        if c > max_c:
            max_c = c
        if c < min_c:
            min_c = c

    return (total, max_c, min_c)
