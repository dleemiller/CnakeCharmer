"""
Maximum product subarray using dynamic programming.

Keywords: dynamic programming, max product, subarray, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def max_product_subarray(n: int) -> tuple:
    """Find maximum product subarray across chunks of a deterministic array.

    Values: v[i] = ((i * 2654435761) % 11) - 5, giving range [-5, 5].
    Zeros are mapped to 1. Split into chunks of 10, compute max product
    subarray for each chunk, then sum results.

    Args:
        n: Total sequence length (should be divisible by 10).

    Returns:
        Tuple of (total_sum_of_max_products, count_of_positive_chunks).
    """
    chunk_size = 10
    k = n // chunk_size
    total = 0
    positive_count = 0

    for chunk in range(k):
        offset = chunk * chunk_size

        # Generate chunk values, skip zeros by mapping 0 -> 1
        vals = [0] * chunk_size
        for i in range(chunk_size):
            v = ((offset + i) * 2654435761) % 11 - 5
            if v == 0:
                v = 1
            vals[i] = v

        # Track max and min product ending at each position
        max_prod = vals[0]
        min_prod = vals[0]
        best = vals[0]

        for i in range(1, chunk_size):
            v = vals[i]
            cand1 = max_prod * v
            cand2 = min_prod * v
            # new_max
            new_max = v
            if cand1 > new_max:
                new_max = cand1
            if cand2 > new_max:
                new_max = cand2
            # new_min
            new_min = v
            if cand1 < new_min:
                new_min = cand1
            if cand2 < new_min:
                new_min = cand2

            max_prod = new_max
            min_prod = new_min
            if max_prod > best:
                best = max_prod

        total += best
        if best > 0:
            positive_count += 1

    return (total, positive_count)
