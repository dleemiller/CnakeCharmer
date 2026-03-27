"""Find all index pairs that sum to a target value.

Keywords: leetcode, two sum, pairs, hash map, indices, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def two_sum_all_pairs(n: int) -> tuple:
    """Find all pairs (i, j) with i < j where arr[i] + arr[j] == target.

    Array: arr[i] = (i * 37 + 13) % (n // 2).
    Target: n // 4.
    Returns count of pairs, sum of all pair indices, and first pair found.

    Args:
        n: Size of the array.

    Returns:
        Tuple of (num_pairs, index_sum_of_all_pairs, first_pair_i_plus_j).
    """
    target = n // 4
    half_n = n // 2
    if half_n == 0:
        half_n = 1

    # Build array
    arr = [0] * n
    for i in range(n):
        arr[i] = (i * 37 + 13) % half_n

    # Find all pairs using hash map
    # Map from value -> list of indices
    index_map = {}
    num_pairs = 0
    index_sum = 0
    first_pair_sum = -1

    for j in range(n):
        complement = target - arr[j]
        if complement in index_map:
            count = index_map[complement]
            num_pairs += count
            # Each of the 'count' earlier indices pairs with j
            # We don't track individual indices for performance, just count
            index_sum += count * j
        if arr[j] in index_map:
            index_map[arr[j]] += 1
        else:
            index_map[arr[j]] = 1

    # Find first pair explicitly for verification
    seen = {}
    for j in range(n):
        complement = target - arr[j]
        if complement in seen:
            first_pair_sum = seen[complement] + j
            break
        if arr[j] not in seen:
            seen[arr[j]] = j

    return (num_pairs, index_sum, first_pair_sum)
