"""Convert probabilities to Phred scores and accumulate.

Keywords: numerical, probability, phred, log10, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(48271, 200000, 1e-6))
def probability_to_phred_sum(seed: int, samples: int, floor: float) -> float:
    """Accumulate synthetic Phred scores."""
    total = 0.0
    state = seed & 0x7FFFFFFF
    for _ in range(samples):
        state = (state * 48271) % 2147483647
        p = floor + (1.0 - floor) * (state / 2147483647.0)
        total += -10.0 * math.log10(p)
    return total
