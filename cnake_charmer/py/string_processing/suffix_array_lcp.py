"""
Build suffix array + LCP array and return sum of LCP values.

Keywords: string processing, suffix array, lcp, longest common prefix, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def suffix_array_lcp(n: int) -> int:
    """Build suffix array and LCP array, return sum of LCP values.

    String: s[i] = chr(65 + (i * 7 + 3) % 4) (uses only 4 chars for more LCP overlap).

    Args:
        n: Length of the string.

    Returns:
        Sum of all values in the LCP array.
    """
    if n <= 0:
        return 0

    s = "".join(chr(65 + (i * 7 + 3) % 4) for i in range(n))

    # Build suffix array (naive O(n^2 log n))
    sa = sorted(range(n), key=lambda idx: s[idx:])

    # Build rank array
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i

    # Build LCP array using Kasai's algorithm
    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    total = 0
    for val in lcp:
        total += val

    return total
