"""Prefix sum (accumulate) on int and double arrays.

Keywords: numerical, prefix sum, accumulate, generic, fused type, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_accumulate(n: int) -> float:
    """Compute prefix sums of int and double arrays, return last elements summed.

    int_arr[i] = (i * 23 + 5) % 509
    double_arr[i] = ((i * 29 + 11) % 601) / 13.0

    Args:
        n: Number of elements in each array.

    Returns:
        Last element of int prefix sum + last element of double prefix sum.
    """
    # Int prefix sum
    int_acc = 0
    for i in range(n):
        int_acc += (i * 23 + 5) % 509

    # Double prefix sum
    dbl_acc = 0.0
    for i in range(n):
        dbl_acc += ((i * 29 + 11) % 601) / 13.0

    return float(int_acc) + dbl_acc
