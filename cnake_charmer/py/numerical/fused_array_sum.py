"""Sum arrays using generic typed helpers for int and double, return combined sum.

Keywords: numerical, array, sum, generic, fused type, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_array_sum(n: int) -> float:
    """Sum an int array and a double array, return combined sum.

    Generates arrays deterministically: int_arr[i] = (i * 7 + 3) % 1000,
    double_arr[i] = (i * 11 + 5) % 1000 / 10.0.

    Args:
        n: Number of elements in each array.

    Returns:
        Combined sum as float (int_sum + double_sum).
    """
    int_sum = 0
    for i in range(n):
        int_sum += (i * 7 + 3) % 1000

    double_sum = 0.0
    for i in range(n):
        double_sum += ((i * 11 + 5) % 1000) / 10.0

    return float(int_sum) + double_sum
