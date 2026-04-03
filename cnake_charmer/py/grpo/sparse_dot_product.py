"""Sparse vector dot product using index-value pairs.

Keywords: grpo, numerical, sparse, linear algebra, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def sparse_dot_product(n: int) -> tuple:
    """Dot product of two sparse vectors stored as sorted (index, value) lists.

    Generates two sparse vectors with ~10% density and computes their dot product
    using a merge-join approach.

    Returns (dot_product, number of matching indices, operations_count).

    Args:
        n: Dimension of the full vector space.

    Returns:
        Tuple of (dot_product, num_matches, ops).
    """
    # Generate sparse vectors (~10% nonzero)
    indices_a = []
    values_a = []
    indices_b = []
    values_b = []

    for i in range(n):
        if (i * 7 + 3) % 10 == 0:
            indices_a.append(i)
            values_a.append((i * 13 + 1) % 100 * 0.01)
        if (i * 11 + 5) % 10 == 0:
            indices_b.append(i)
            values_b.append((i * 17 + 2) % 100 * 0.01)

    # Merge-join dot product
    dot = 0.0
    matches = 0
    ops = 0
    ia = 0
    ib = 0
    len_a = len(indices_a)
    len_b = len(indices_b)

    while ia < len_a and ib < len_b:
        ops += 1
        if indices_a[ia] == indices_b[ib]:
            dot += values_a[ia] * values_b[ib]
            matches += 1
            ia += 1
            ib += 1
        elif indices_a[ia] < indices_b[ib]:
            ia += 1
        else:
            ib += 1

    return (round(dot, 6), matches, ops)
