"""
Naive suffix array construction.

Keywords: string processing, suffix array, sorting, naive, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def suffix_array_naive(n: int) -> int:
    """Build a suffix array of a deterministic string using naive O(n^2 log n) sort,
    and return the sum of the first 100 suffix positions.

    String: s[i] = chr(65 + (i*7+3) % 26) for i in range(n).

    Args:
        n: Length of the string.

    Returns:
        Sum of first min(100, n) suffix array positions.
    """
    s = "".join(chr(65 + (i * 7 + 3) % 26) for i in range(n))

    # Build suffix array by sorting indices by suffix
    sa = sorted(range(n), key=lambda i: s[i:])

    # Sum first 100 positions
    limit = min(100, n)
    total = 0
    for k in range(limit):
        total += sa[k]

    return total
