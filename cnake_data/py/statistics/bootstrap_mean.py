"""
Bootstrap mean variance via deterministic resampling.

Keywords: statistics, bootstrap, resampling, variance, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def bootstrap_mean(n: int) -> float:
    """Compute variance of k=1000 bootstrap means using deterministic resampling.

    Dataset: v[i] = (i*17+5) % 1000 / 10.0
    For each bootstrap iteration b in range(k):
        resample index j -> idx = (j * b * 31 + 7) % n
        compute mean of resampled values
    Return variance of the k bootstrap means.

    Args:
        n: Size of the dataset.

    Returns:
        Variance of the bootstrap means.
    """
    k = 1000

    # Build dataset
    data = [0.0] * n
    for i in range(n):
        data[i] = ((i * 17 + 5) % 1000) / 10.0

    # Compute bootstrap means
    means = [0.0] * k
    for b in range(k):
        total = 0.0
        for j in range(n):
            idx = (j * b * 31 + 7) % n
            total += data[idx]
        means[b] = total / n

    # Compute variance of means
    sum_m = 0.0
    for b in range(k):
        sum_m += means[b]
    avg = sum_m / k

    var = 0.0
    for b in range(k):
        diff = means[b] - avg
        var += diff * diff
    var /= k

    return var
