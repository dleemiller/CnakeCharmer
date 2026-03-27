"""Burrows-Wheeler Transform returning character codes and primary index.

Keywords: string processing, burrows-wheeler, bwt, transform, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def burrows_wheeler(n: int) -> tuple:
    """Compute BWT of a deterministic string.

    String: s[i] = chr(65 + (i*7+3) % 4) for i in 0..n-1, with null sentinel.

    Args:
        n: Length of input string (before sentinel).

    Returns:
        Tuple of (first_char_code, last_char_code, primary_index).
        first/last are ord() of first/last char in BWT output.
        primary_index is the position of the original string in sorted rotations.
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
    bwt_first = ord(s[(indices[0] - 1) % length])
    bwt_last = ord(s[(indices[length - 1] - 1) % length])

    # Primary index: where does rotation 0 appear?
    primary_index = 0
    for i in range(length):
        if indices[i] == 0:
            primary_index = i
            break

    return (bwt_first, bwt_last, primary_index)
