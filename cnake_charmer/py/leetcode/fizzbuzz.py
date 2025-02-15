"""
FizzBuzz Implementation in Python.

This module provides a simple FizzBuzz example as part of the living dataset.

Keywords: fizzbuzz, leetcode, python, benchmark, example

"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def fizzbuzz(n: int) -> list[str]:
    """Generate the FizzBuzz sequence for numbers from 1 to n.

    Args:
        n (int): The upper limit of numbers to process.

    Returns:
        List[str]: A list where multiples of 3 are replaced with 'Fizz',
        multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print("\n".join(fizzbuzz(n)))
