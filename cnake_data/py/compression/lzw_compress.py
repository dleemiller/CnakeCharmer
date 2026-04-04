"""LZW compression dictionary size computation.

Keywords: compression, lzw, dictionary, encoding, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def lzw_compress(n: int) -> int:
    """Count dictionary entries created during LZW compression.

    Input string: s[i] = chr(65 + (i*7+3)%4) (characters A-D).
    Standard LZW: start with single-char dictionary, build up multi-char
    entries as new patterns are found.

    Args:
        n: Length of input string.

    Returns:
        Total dictionary size after compression.
    """
    # Initialize dictionary with single characters
    dictionary = {}
    for c in range(4):
        dictionary[chr(65 + c)] = c

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
