"""Sum of absolute values for int, long, and double arrays.

Demonstrates type-checking pattern where different types
use different abs implementations.

Keywords: numerical, fused type, type check, absolute value, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_type_check_abs(n: int) -> float:
    """Sum absolute values of int, long, and double arrays.

    Args:
        n: Number of elements per array.

    Returns:
        Total of all three abs-sums.
    """
    # Int abs sum (values centered around 0)
    int_total = 0
    for i in range(n):
        val = (i * 37 + 13) % 997 - 498
        int_total += abs(val)

    # Long abs sum
    long_total = 0
    for i in range(n):
        val = (i * 53 + 29) % 10007 - 5003
        long_total += abs(val)

    # Double abs sum
    dbl_total = 0.0
    for i in range(n):
        val_d = ((i * 41 + 7) % 1009 - 504) / 13.0
        dbl_total += abs(val_d)

    return float(int_total) + float(long_total) + dbl_total
