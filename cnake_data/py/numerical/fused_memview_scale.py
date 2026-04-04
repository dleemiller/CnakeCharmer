"""In-place scale of float and double arrays using a generic helper.

Demonstrates fused-type in-place memoryview scaling.

Keywords: numerical, fused type, memoryview, scale, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_memview_scale(n: int) -> float:
    """Scale float and double arrays in-place, return sums.

    Args:
        n: Number of elements per array.

    Returns:
        Sum of scaled float array + sum of scaled double array.
    """
    factor = 2.5

    # Float array (simulated as Python float)
    float_arr = [0.0] * n
    for i in range(n):
        float_arr[i] = ((i * 31 + 17) % 503) / 11.0
    for i in range(n):
        float_arr[i] *= factor

    float_total = 0.0
    for i in range(n):
        float_total += float_arr[i]

    # Double array
    dbl_arr = [0.0] * n
    for i in range(n):
        dbl_arr[i] = ((i * 43 + 19) % 607) / 13.0
    for i in range(n):
        dbl_arr[i] *= factor

    dbl_total = 0.0
    for i in range(n):
        dbl_total += dbl_arr[i]

    return float_total + dbl_total
