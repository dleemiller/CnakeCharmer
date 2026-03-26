"""Shannon entropy computation.

Keywords: compression, entropy, shannon, information theory, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def entropy(n: int) -> float:
    """Compute Shannon entropy of a deterministic symbol sequence.

    Generates n symbols from alphabet of 26 using s[i] = (i * 7 + 3) % 26,
    computes character frequencies, then calculates Shannon entropy.

    Args:
        n: Number of symbols.

    Returns:
        Shannon entropy in bits.
    """
    freq = [0] * 26

    for i in range(n):
        freq[(i * 7 + 3) % 26] += 1

    result = 0.0
    for i in range(26):
        if freq[i] > 0:
            p = freq[i] / n
            result -= p * math.log(p)

    return result
