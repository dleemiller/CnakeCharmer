"""
Count inversions in a deterministic permutation using merge sort.

Keywords: algorithms, inversions, merge sort, counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def count_inversions(n: int) -> int:
    """Count the number of inversions in a deterministic permutation.

    The permutation is arr[i] = (i * 7 + 13) % n. An inversion is a pair
    (i, j) where i < j but arr[i] > arr[j]. Uses merge sort counting.

    Args:
        n: Size of the permutation.

    Returns:
        Tuple of (total inversions, positional checksum of sorted array).
    """
    arr = [(i * 7 + 13) % n for i in range(n)]

    def merge_count(lst):
        if len(lst) <= 1:
            return lst, 0
        mid = len(lst) // 2
        left, left_inv = merge_count(lst[:mid])
        right, right_inv = merge_count(lst[mid:])

        merged = []
        inversions = left_inv + right_inv
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    sorted_arr, total = merge_count(arr)
    checksum = sum(i * sorted_arr[i] for i in range(n)) % (10**9 + 7)
    return (total, checksum)
