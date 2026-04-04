"""
Total longest common prefix length between consecutive string pairs.

Keywords: string processing, longest common prefix, comparison, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def longest_common_prefix(n: int) -> int:
    """Find the total length of the longest common prefix between all consecutive pairs.

    Strings: s[i] = "".join(chr(65 + (i*j+3)%26) for j in range(20)).

    Args:
        n: Number of strings.

    Returns:
        Sum of LCP lengths for all consecutive pairs.
    """
    # Generate strings
    strings = ["".join(chr(65 + (i * j + 3) % 26) for j in range(20)) for i in range(n)]

    total = 0
    max_lcp = 0
    for i in range(n - 1):
        s1 = strings[i]
        s2 = strings[i + 1]
        lcp = 0
        for k in range(20):
            if s1[k] == s2[k]:
                lcp += 1
            else:
                break
        total += lcp
        if lcp > max_lcp:
            max_lcp = lcp

    return (total, max_lcp)
