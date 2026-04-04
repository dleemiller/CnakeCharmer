"""Evaluate a piecewise linear function via lookup table.

Keywords: lookup table, interpolation, piecewise linear, container, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def lookup_table_eval(n: int) -> float:
    """Build a 256-entry lookup table and evaluate n interpolated lookups.

    The table maps [0, 255] to f(x) = sin-like values. Each lookup interpolates
    between adjacent entries. Returns the sum of all evaluated values.

    Args:
        n: Number of lookups to perform.

    Returns:
        Sum of all interpolated results.
    """
    import math

    # Build table
    table_size = 256
    table = [0.0] * table_size
    for i in range(table_size):
        table[i] = math.sin(2.0 * math.pi * i / table_size) * 100.0

    total = 0.0
    for i in range(n):
        # Generate a fractional index in [0, 255]
        h = ((i * 2654435761 + 17) >> 4) & 0xFFFF
        frac_idx = (h % 25500) / 100.0  # [0, 255)

        # Integer part and fraction
        idx = int(frac_idx)
        frac = frac_idx - idx

        if idx >= table_size - 1:
            total += table[table_size - 1]
        else:
            # Linear interpolation
            total += table[idx] * (1.0 - frac) + table[idx + 1] * frac

    return total
