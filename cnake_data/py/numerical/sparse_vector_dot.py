"""Dot product of two sparse vectors stored as index-value pairs.

Keywords: sparse vector, dot product, container, sequence protocol, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def sparse_vector_dot(n: int) -> float:
    """Create two sparse vectors of dimension n with ~10% density, compute dot product.

    Args:
        n: Vector dimension.

    Returns:
        Dot product of the two sparse vectors.
    """
    # Build sparse vector A: store (index, value) pairs
    indices_a = []
    values_a = []
    for i in range(n):
        h = ((i * 2654435761 + 1) >> 8) & 0xFF
        if h < 26:  # ~10% density
            indices_a.append(i)
            values_a.append(((h * 31 + 7) % 200 - 100) / 10.0)

    # Build sparse vector B
    indices_b = []
    values_b = []
    for i in range(n):
        h = ((i * 1103515245 + 3) >> 8) & 0xFF
        if h < 26:
            indices_b.append(i)
            values_b.append(((h * 37 + 11) % 200 - 100) / 10.0)

    # Dot product via merge of sorted index lists
    size_a = len(indices_a)
    size_b = len(indices_b)
    ia = 0
    ib = 0
    dot = 0.0

    while ia < size_a and ib < size_b:
        if indices_a[ia] == indices_b[ib]:
            dot += values_a[ia] * values_b[ib]
            ia += 1
            ib += 1
        elif indices_a[ia] < indices_b[ib]:
            ia += 1
        else:
            ib += 1

    return dot
