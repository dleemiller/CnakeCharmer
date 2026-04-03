"""Prime testing with while-loop trial division.

Tests primality using a while loop instead of for loop, which can be
more natural for Cython optimization since it avoids Python range objects.

Keywords: prime, while_loop, trial_division, primality, math
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def count_primes_while(n: int) -> int:
    """Count the number of primes up to n using while-loop trial division.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Number of primes in [2, n].
    """
    count = 0
    for num in range(2, n + 1):
        if _is_prime_while(num):
            count += 1
    return count


def _is_prime_while(n: int) -> bool:
    """Test primality using while loop."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    n_sqrt = int(math.sqrt(n)) + 1
    while i < n_sqrt:
        if n % i == 0:
            return False
        i += 2
    return True
