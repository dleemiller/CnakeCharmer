"""Find length of longest repeated substring using suffix array and LCP.

Keywords: suffix array, lcp, longest repeated substring, string, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def longest_repeated_substring(n: int) -> int:
    """Find the length of the longest repeated substring.

    String is generated as: s[i] = chr(65 + (i*7+3) % 26).
    Uses suffix array construction (simple sort) + LCP array.

    Args:
        n: Length of the string.

    Returns:
        Length of longest substring that appears at least twice.
    """
    # Generate string
    s = "".join(chr(65 + (i * 7 + 3) % 26) for i in range(n))

    # Build suffix array by sorting suffix indices
    sa = list(range(n))
    sa.sort(key=lambda i: s[i:])

    # Compute LCP between adjacent suffixes in sorted order
    best = 0
    for i in range(1, n):
        # Compare s[sa[i-1]:] and s[sa[i]:]
        a = sa[i - 1]
        b = sa[i]
        lcp_len = 0
        while a + lcp_len < n and b + lcp_len < n and s[a + lcp_len] == s[b + lcp_len]:
            lcp_len += 1
        if lcp_len > best:
            best = lcp_len

    return best
