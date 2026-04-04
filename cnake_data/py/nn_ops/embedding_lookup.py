"""Embedding lookup on a float tensor.

Lookup n embeddings of dim=64 from a table of 1000 entries.

Keywords: embedding, lookup, neural network, tensor, f32, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def embedding_lookup(n: int) -> float:
    """Lookup n embeddings from table, return sum of all values.

    Args:
        n: Number of embeddings to look up.

    Returns:
        Sum of all looked-up embedding values.
    """
    vocab_size = 1000
    dim = 64

    # Build embedding table [vocab_size x dim]
    table = []
    for v in range(vocab_size):
        row = []
        for d in range(dim):
            row.append(math.sin((v * dim + d) * 0.01) * 0.1)
        table.append(row)

    # Compute indices
    total = 0.0
    for i in range(n):
        idx = (i * 7 + 3) % vocab_size
        row = table[idx]
        for d in range(dim):
            total += row[d]

    return total
