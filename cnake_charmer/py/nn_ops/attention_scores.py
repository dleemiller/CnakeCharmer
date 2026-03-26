"""Scaled dot-product attention scores.

Keywords: attention, transformer, dot product, neural network, self-attention
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def attention_scores(n: int) -> float:
    """Compute scaled dot-product attention scores Q*K^T/sqrt(d).

    Q[i][j] = sin(i * 0.1 + j * 0.01), K[i][j] = cos(i * 0.1 + j * 0.01).
    d = 64.

    Args:
        n: Number of queries (and keys).

    Returns:
        Sum of all attention scores.
    """
    d = 64
    inv_sqrt_d = 1.0 / math.sqrt(d)
    total = 0.0
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                q = math.sin(i * 0.1 + k * 0.01)
                kv = math.cos(j * 0.1 + k * 0.01)
                dot += q * kv
            total += dot * inv_sqrt_d
    return total
