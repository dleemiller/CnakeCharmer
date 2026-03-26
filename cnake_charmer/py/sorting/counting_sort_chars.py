"""Count characters and compute frequency-rank weighted sum.

Keywords: sorting, counting sort, character frequency, ranking, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def counting_sort_chars(n: int) -> int:
    """Sort characters by frequency and compute weighted rank sum.

    Generates s[i] = chr(65 + (i * 7 + 3) % 26), counts character
    frequencies, sorts by frequency (descending), then computes the
    sum of freq * rank (1-indexed) for all characters.

    Args:
        n: Length of the string.

    Returns:
        Sum of freq * rank for all characters, ranked by descending frequency.
    """
    # Count character frequencies
    counts = [0] * 26
    for i in range(n):
        counts[(i * 7 + 3) % 26] += 1

    # Sort frequencies descending
    freq_list = []
    for i in range(26):
        if counts[i] > 0:
            freq_list.append(counts[i])
    freq_list.sort(reverse=True)

    # Compute sum of freq * rank
    total = 0
    for rank, freq in enumerate(freq_list, 1):
        total += freq * rank

    return total
