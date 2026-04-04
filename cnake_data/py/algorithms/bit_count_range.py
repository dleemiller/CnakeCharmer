"""Count total set bits across all integers in a range.

Keywords: algorithms, bit counting, popcount, bitwise, range, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def bit_count_range(n: int) -> tuple:
    """Count set bits across all integers in [0, n) and compute related statistics.

    For each integer in range, counts its set bits. Returns the total set bits,
    the number of integers with exactly 10 set bits, and the maximum run of
    consecutive integers with an odd popcount.

    Args:
        n: Upper bound of the range (exclusive).

    Returns:
        Tuple of (total_bits, count_with_10_bits, max_odd_run).
    """
    total_bits = 0
    count_with_10 = 0
    max_odd_run = 0
    current_odd_run = 0

    for i in range(n):
        # Count set bits manually
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1

        total_bits += count

        if count == 10:
            count_with_10 += 1

        if count & 1:
            current_odd_run += 1
            if current_odd_run > max_odd_run:
                max_odd_run = current_odd_run
        else:
            current_odd_run = 0

    return (total_bits, count_with_10, max_odd_run)
