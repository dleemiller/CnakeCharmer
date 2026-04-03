"""Numerically stable log-sum-exp using the max trick.

Keywords: numerical, log_sum_exp, exponential, logarithm, stability, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def log_sum_exp(n: int) -> tuple:
    """Compute log-sum-exp over deterministic log-probability arrays.

    Generates n log-probability values deterministically, then computes
    log(sum(exp(lnp))) using the max trick for numerical stability.
    Repeats with 5 different offsets and accumulates results.

    Args:
        n: Number of log-probability values per batch.

    Returns:
        Tuple of (accumulated_result, last_max_val).
    """
    accum = 0.0
    max_val = 0.0

    for offset in range(5):
        # Generate deterministic log-probabilities
        # Find max for numerical stability
        max_lnp = -1e308
        for i in range(n):
            val = -((i * 7 + 3 + offset * 13) % 100) / 10.0
            if val > max_lnp:
                max_lnp = val

        # Compute sum of exp(lnp - max)
        sum_exp = 0.0
        for i in range(n):
            val = -((i * 7 + 3 + offset * 13) % 100) / 10.0
            sum_exp += math.exp(val - max_lnp)

        result = math.log(sum_exp) + max_lnp
        accum += result
        max_val = max_lnp

    return (accum, max_val)
