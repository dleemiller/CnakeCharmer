"""Compute Levenshtein distances and count strings within threshold.

Keywords: levenshtein, edit distance, string matching, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def levenshtein_automaton(n: int) -> int:
    """Compute Levenshtein distance of n generated strings against HELLO.

    Strings are 5-char from 16-char alphabet (A-P), generated via
    multiplicative hash. Counts strings within edit distance 2.

    Args:
        n: Number of strings to check.

    Returns:
        Count of strings within edit distance 2.
    """
    target = "HELLO"
    tlen = 5
    count = 0

    for i in range(n):
        # Generate string via multiplicative hash for good distribution
        v = i * 2654435761
        s0 = chr(65 + (v >> 0) % 16)
        s1 = chr(65 + (v >> 4) % 16)
        s2 = chr(65 + (v >> 8) % 16)
        s3 = chr(65 + (v >> 12) % 16)
        s4 = chr(65 + (v >> 16) % 16)
        chars = [s0, s1, s2, s3, s4]

        # Compute edit distance with early termination
        prev = [0, 1, 2, 3, 4, 5]
        curr = [0] * 6

        too_far = False
        for si in range(5):
            curr[0] = si + 1
            row_min = curr[0]
            for ti in range(tlen):
                cost = 0 if chars[si] == target[ti] else 1
                ins = curr[ti] + 1
                dele = prev[ti + 1] + 1
                sub = prev[ti] + cost
                val = ins
                if dele < val:
                    val = dele
                if sub < val:
                    val = sub
                curr[ti + 1] = val
                if val < row_min:
                    row_min = val
            if row_min > 2:
                too_far = True
                break
            prev, curr = curr, prev

        if not too_far and prev[tlen] <= 2:
            count += 1

    return count
