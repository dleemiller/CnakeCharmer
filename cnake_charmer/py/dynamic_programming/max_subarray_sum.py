"""
Sum of maximum subarray sums using Kadane's algorithm.

Keywords: dynamic programming, kadane, max subarray, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def max_subarray_sum(n: int) -> int:
    """Find sum of max subarray sums for k=n/100 subarrays of length 100.

    Sequence: v[i] = ((i*17+5)%201) - 100, giving values in [-100, 100].
    Split into k=n//100 chunks of 100 elements each.
    Apply Kadane's algorithm to each chunk and sum the results.

    Args:
        n: Total sequence length (should be divisible by 100).

    Returns:
        Sum of maximum subarray sums across all chunks.
    """
    chunk_size = 100
    k = n // chunk_size
    total = 0

    for chunk in range(k):
        offset = chunk * chunk_size
        max_ending_here = 0
        max_so_far = -101  # smaller than any possible element

        for i in range(chunk_size):
            val = ((offset + i) * 17 + 5) % 201 - 100
            max_ending_here += val
            if max_ending_here < val:
                max_ending_here = val
            if max_ending_here > max_so_far:
                max_so_far = max_ending_here

        total += max_so_far

    return total
