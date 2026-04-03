"""
Huber loss between two deterministic sequences.

Computes the Huber loss (smooth L1 loss) with delta=0.0125 between two
sequences of length n. Returns (average_loss, max_residual).

Keywords: statistics, huber, loss, robust, regression, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def huber_loss(n: int) -> tuple:
    """Compute Huber loss between two deterministic sequences of length n.

    Sequences are generated deterministically:
        x[i] = ((i * 7 + 3) % 1000) / 500.0 - 1.0
        y[i] = ((i * 13 + 7) % 1000) / 500.0 - 1.0

    Args:
        n: Length of the sequences.

    Returns:
        Tuple of (average huber loss, max absolute residual).
    """
    delta = 0.0125
    res_sum = 0.0
    max_res = 0.0

    for i in range(n):
        xi = ((i * 7 + 3) % 1000) / 500.0 - 1.0
        yi = ((i * 13 + 7) % 1000) / 500.0 - 1.0
        res = abs(xi - yi)

        if res > max_res:
            max_res = res

        if res <= delta:
            res_sum += (res * res) / 2.0
        else:
            res_sum += delta * (res - delta / 2.0)

    return (res_sum / n, max_res)
