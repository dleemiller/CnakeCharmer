"""Arithmetic coding interval width computation.

Keywords: compression, arithmetic coding, entropy, frequency, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def arithmetic_coding_freq(n: int) -> float:
    """Compute sum of log interval widths for arithmetic coding of n symbols.

    Symbols: s[i] = (i*7+3) % 4 (4-symbol alphabet).
    Computes cumulative frequency table, then simulates arithmetic coding
    by tracking interval narrowing. Returns sum of log2(interval_width)
    sampled every 100 symbols to avoid underflow.

    Args:
        n: Number of symbols.

    Returns:
        Sum of log interval widths (entropy-like measure).
    """
    # Count frequencies
    freq = [0] * 4
    for i in range(n):
        sym = (i * 7 + 3) % 4
        freq[sym] += 1

    # Compute cumulative frequencies (scaled to total n)
    total = n
    cum = [0] * 5
    for i in range(4):
        cum[i + 1] = cum[i] + freq[i]

    # Simulate arithmetic coding with periodic rescaling
    log_width_sum = 0.0
    log_width = 0.0  # log2 of current interval width (starts at log2(1)=0)

    for i in range(n):
        sym = (i * 7 + 3) % 4
        # Interval narrows by factor freq[sym]/total
        log_width += math.log(freq[sym] / total)

        # Sample every 100 symbols
        if (i + 1) % 100 == 0:
            log_width_sum += log_width
            log_width = 0.0

    # Add remaining
    log_width_sum += log_width

    return log_width_sum
