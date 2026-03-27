"""Find min and max of int and double arrays, return combined result.

Keywords: numerical, min, max, array, generic, fused type, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_minmax(n: int) -> float:
    """Find min/max of int and double arrays, return combined result.

    int_arr[i] = (i * 31 + 17) % 100003
    double_arr[i] = ((i * 37 + 11) % 99991) / 7.0

    Args:
        n: Number of elements in each array.

    Returns:
        min_int + max_int + min_double + max_double as float.
    """
    min_int = (0 * 31 + 17) % 100003
    max_int = min_int
    for i in range(1, n):
        val = (i * 31 + 17) % 100003
        if val < min_int:
            min_int = val
        if val > max_int:
            max_int = val

    first_dbl = ((0 * 37 + 11) % 99991) / 7.0
    min_dbl = first_dbl
    max_dbl = first_dbl
    for i in range(1, n):
        val_d = ((i * 37 + 11) % 99991) / 7.0
        if val_d < min_dbl:
            min_dbl = val_d
        if val_d > max_dbl:
            max_dbl = val_d

    return float(min_int) + float(max_int) + min_dbl + max_dbl
