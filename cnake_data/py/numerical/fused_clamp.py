"""Clamp array values to a range and return the sum.

Keywords: numerical, clamp, array, generic, fused type, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_clamp(n: int) -> float:
    """Clamp int and double arrays to [lo, hi], return combined sum.

    int_arr[i] = (i * 47 + 13) % 2003 - 1000
    double_arr[i] = ((i * 53 + 7) % 1999 - 999) / 3.0
    Int clamp: [−200, 200]. Double clamp: [−50.0, 50.0].

    Args:
        n: Number of elements in each array.

    Returns:
        Sum of clamped int array + sum of clamped double array.
    """
    int_sum = 0
    for i in range(n):
        val = (i * 47 + 13) % 2003 - 1000
        if val < -200:
            val = -200
        elif val > 200:
            val = 200
        int_sum += val

    dbl_sum = 0.0
    for i in range(n):
        val_d = ((i * 53 + 7) % 1999 - 999) / 3.0
        if val_d < -50.0:
            val_d = -50.0
        elif val_d > 50.0:
            val_d = 50.0
        dbl_sum += val_d

    return float(int_sum) + dbl_sum
