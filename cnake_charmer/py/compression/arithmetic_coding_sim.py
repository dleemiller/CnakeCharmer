"""Simulate arithmetic coding interval narrowing with integer precision.

Keywords: compression, arithmetic coding, interval, simulation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def arithmetic_coding_sim(n: int) -> tuple:
    """Simulate arithmetic coding with integer interval narrowing.

    Uses a 4-symbol alphabet with fixed frequencies. Processes n symbols,
    tracking the integer interval [low, high) with periodic rescaling
    to prevent underflow.

    Args:
        n: Number of symbols to encode.

    Returns:
        Tuple of (final_low_mod, final_high_mod, num_rescales).
    """
    # 4-symbol alphabet with fixed cumulative frequencies
    # Symbol i has probability proportional to (i + 1)
    # Total = 1 + 2 + 3 + 4 = 10
    cum = [0, 1, 3, 6, 10]
    total = 10

    precision = 30  # bits
    full_range = 1 << precision
    half = full_range >> 1
    quarter = full_range >> 2

    low = 0
    high = full_range - 1
    num_rescales = 0

    for i in range(n):
        sym = (i * 13 + 7) % 4

        rng = high - low + 1
        high = low + (rng * cum[sym + 1]) // total - 1
        low = low + (rng * cum[sym]) // total

        # Rescale when top bits match
        while True:
            if high < half:
                # Both in lower half, shift out MSB 0
                low = low << 1
                high = (high << 1) | 1
                num_rescales += 1
            elif low >= half:
                # Both in upper half, shift out MSB 1
                low = (low - half) << 1
                high = ((high - half) << 1) | 1
                num_rescales += 1
            elif low >= quarter and high < 3 * quarter:
                # Middle range expansion
                low = (low - quarter) << 1
                high = ((high - quarter) << 1) | 1
                num_rescales += 1
            else:
                break

    # Return modded values to keep them reasonable
    mod_val = 1000000007
    return (low % mod_val, high % mod_val, num_rescales)
