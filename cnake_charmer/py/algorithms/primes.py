"""
Primes Module
-------------

This module provides a simple prime number generator that uses trial division
to compute a specified number of prime numbers.
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def primes(nb_primes: int) -> list[int]:
    """Generate a list of prime numbers.

    This function generates prime numbers using trial division until the desired
    count of prime numbers is reached.

    Args:
        nb_primes (int): The number of prime numbers to generate.

    Returns:
        list[int]: A list containing the first nb_primes prime numbers.
    """
    primes_list: list[int] = []
    n: int = 2

    while len(primes_list) < nb_primes:
        for prime in primes_list:
            if n % prime == 0:
                break
        else:
            primes_list.append(n)
        n += 1

    return primes_list
