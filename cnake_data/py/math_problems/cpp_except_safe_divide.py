"""Safe integer division with exception handling for division by zero.

Keywords: math, division, exception, ValueError, safe divide, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _safe_divide(a: int, b: int) -> int:
    """Divide a by b (truncation toward zero), raising ValueError if b is zero."""
    if b == 0:
        raise ValueError("division by zero")
    # Match C integer division (truncation toward zero)
    return int(a / b)


@python_benchmark(args=(500000,))
def cpp_except_safe_divide(n: int) -> tuple:
    """Perform n integer divisions with b in [-10, 10] (b=0 raises ValueError).

    a[i] = (i * 2654435761) % 1000 - 500
    b[i] = (i * 1103515245) % 21 - 10  (range [-10, 10], includes 0)
    Catch ValueError for b==0, count errors.

    Args:
        n: Number of division operations.

    Returns:
        Tuple of (sum_of_valid_results, error_count).
    """
    sum_valid = 0
    error_count = 0
    mask = 0xFFFFFFFF
    for i in range(n):
        a = ((i * 2654435761) & mask) % 1000 - 500
        b = ((i * 1103515245) & mask) % 21 - 10
        try:
            sum_valid += _safe_divide(a, b)
        except ValueError:
            error_count += 1
    return (sum_valid, error_count)
