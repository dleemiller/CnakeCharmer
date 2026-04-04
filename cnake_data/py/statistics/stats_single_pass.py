"""Single-pass statistics computation.

Computes min, max, sum, mean, and standard deviation in a single
iteration through the data, avoiding multiple passes.

Keywords: statistics, single_pass, mean, stddev, min_max, efficient
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def stats_single_pass(n: int) -> dict:
    """Compute statistics on a generated list of n floats in one pass.

    Args:
        n: Number of data points.

    Returns:
        Dictionary with len, min, max, sum, mean, pstdev.
    """
    # Generate data deterministically
    data = []
    x = 1.0
    for _i in range(n):
        x = (x * 1103515245 + 12345) % (2**31)
        data.append(x / (2**31) * 100.0)

    min_val = data[0]
    max_val = data[0]
    total = 0.0
    sum_sq = 0.0
    for v in data:
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v
        total += v
        sum_sq += v * v

    length = len(data)
    mean = total / length
    pstdev = ((sum_sq / length) - (mean * mean)) ** 0.5

    return {
        "len": length,
        "min": min_val,
        "max": max_val,
        "sum": total,
        "mean": mean,
        "pstdev": pstdev,
    }
