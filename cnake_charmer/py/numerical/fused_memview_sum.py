"""Sum arrays of int and double using a generic helper.

Demonstrates fused-type memoryview parameter pattern.
Creates int and double arrays, sums each, returns total.

Keywords: numerical, fused type, memoryview, sum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_memview_sum(n: int) -> float:
    """Sum int and double arrays using generic approach.

    Args:
        n: Number of elements per array.

    Returns:
        Sum of int array + sum of double array.
    """
    # Int array sum
    int_total = 0
    for i in range(n):
        int_total += (i * 37 + 13) % 997

    # Double array sum
    dbl_total = 0.0
    for i in range(n):
        dbl_total += ((i * 41 + 7) % 1009) / 17.0

    return float(int_total) + dbl_total
