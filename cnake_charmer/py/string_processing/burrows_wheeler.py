"""Burrows-Wheeler Transform and byte sum of output.

Keywords: string processing, burrows-wheeler, bwt, transform, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def burrows_wheeler(n: int) -> int:
    """Compute BWT of a deterministic string and return sum of output bytes.

    String: s[i] = chr(65 + (i*7+3) % 4) for i in 0..n-1.

    Args:
        n: Length of input string.

    Returns:
        Sum of byte values of BWT output.
    """
    s = ""
    for i in range(n):
        s += chr(65 + (i * 7 + 3) % 4)
    s += "\x00"  # sentinel
    length = len(s)

    # Build suffix array via sorted rotations
    indices = list(range(length))
    indices.sort(key=lambda idx: s[idx:] + s[:idx])

    # BWT is the last column: char before each sorted rotation
    total = 0
    for idx in indices:
        total += ord(s[(idx - 1) % length])

    return total
