"""Find all happy numbers up to n.

Keywords: happy, numbers, digit, squares, math, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def happy_numbers(n: int) -> tuple:
    """Count happy numbers from 1 to n and find the largest one.

    A happy number is one that eventually reaches 1 when you repeatedly
    replace it with the sum of the squares of its digits.

    Args:
        n: Upper bound (inclusive) for searching happy numbers.

    Returns:
        Tuple of (count of happy numbers, largest happy number found,
        checksum of all happy numbers mod 10^9+7).
    """
    mod = 10**9 + 7
    count = 0
    last_happy = 0
    checksum = 0

    for num in range(1, n + 1):
        val = num
        seen = set()
        while val != 1 and val not in seen:
            seen.add(val)
            s = 0
            tmp = val
            while tmp > 0:
                d = tmp % 10
                s += d * d
                tmp //= 10
            val = s

        if val == 1:
            count += 1
            last_happy = num
            checksum = (checksum + num) % mod

    return (count, last_happy, checksum)
