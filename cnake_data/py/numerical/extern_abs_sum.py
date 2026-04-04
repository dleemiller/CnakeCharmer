"""Compute sum of absolute values for ints and floats.

Keywords: numerical, extern, abs, fabs, sum, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def extern_abs_sum(n: int) -> float:
    """Compute abs sum for n ints and fabs sum for n doubles.

    Args:
        n: Number of values.

    Returns:
        Combined sum of abs(int) + abs(float) values.
    """
    int_sum = 0
    for i in range(n):
        val = ((i * 2654435761 + 17) & 0x7FFFFFFF) % 10000 - 5000
        int_sum += abs(val)

    float_sum = 0.0
    for i in range(n):
        seed = (i * 1103515245 + 12345) & 0x7FFFFFFF
        fval = (seed % 100000) / 100.0 - 500.0
        float_sum += abs(fval)

    return int_sum + float_sum
