"""
Count occurrences of all 2-character substrings in a deterministic string.

Keywords: string processing, substring counting, algorithm, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def count_substrings(n: int) -> int:
    """Count total occurrences of all unique 2-char substrings.

    Builds a deterministic string s = "".join(chr(65 + (i*7+3) % 26) for i in range(n)),
    then counts all 2-character substring occurrences and returns the total count.

    Args:
        n: Length of the string to generate.

    Returns:
        Total count of all 2-char substring occurrences.
    """
    s = "".join(chr(65 + (i * 7 + 3) % 26) for i in range(n))

    counts = {}
    for i in range(len(s) - 1):
        pair = s[i : i + 2]
        counts[pair] = counts.get(pair, 0) + 1

    total = 0
    for v in counts.values():
        total += v

    return total
