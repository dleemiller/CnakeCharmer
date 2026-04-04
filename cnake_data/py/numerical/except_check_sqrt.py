"""Newton's method square root with error checking.

Keywords: numerical, square root, newton, error handling, except, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _safe_sqrt(x):
    """Compute square root using Newton's method. Returns -1.0 for negative input."""
    if x < 0.0:
        return -1.0
    if x == 0.0:
        return 0.0
    guess = x * 0.5
    for _ in range(30):
        guess = 0.5 * (guess + x / guess)
    return guess


@python_benchmark(args=(100000,))
def except_check_sqrt(n: int) -> float:
    """Compute square roots of n values using Newton's method with error checking.

    Values: val = (i * 67 + 23) % 2003 - 1000 (can be negative).
    If val < 0, skip (safe_sqrt returns -1.0). Otherwise add sqrt to total.
    Count valid results and return total + count.

    Args:
        n: Number of values to process.

    Returns:
        Sum of valid square roots plus count of valid values.
    """
    total = 0.0
    count = 0
    for i in range(n):
        val = ((i * 67 + 23) % 2003) - 1000
        result = _safe_sqrt(float(val))
        if result >= 0.0:
            total += result
            count += 1

    return total + float(count)
