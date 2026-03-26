"""Find majority element using Boyer-Moore voting algorithm.

Keywords: algorithms, moore voting, majority element, counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def moore_voting(n: int) -> int:
    """Find majority element in arr[i] = (i*7+3) % 5, return element + count.

    Uses Boyer-Moore voting algorithm to find candidate, then verifies.

    Args:
        n: Size of the array.

    Returns:
        Sum of majority element value and its count.
    """
    # Phase 1: find candidate
    candidate = 0
    votes = 0
    for i in range(n):
        val = (i * 7 + 3) % 5
        if votes == 0:
            candidate = val
            votes = 1
        elif val == candidate:
            votes += 1
        else:
            votes -= 1

    # Phase 2: verify
    count = 0
    for i in range(n):
        val = (i * 7 + 3) % 5
        if val == candidate:
            count += 1

    return candidate + count
