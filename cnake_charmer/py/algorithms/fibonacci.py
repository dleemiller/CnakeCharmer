from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1e18,))
def fib(n: int) -> list[int]:
    """Return the Fibonacci series up to n as a list."""
    a, b = 0, 1
    result = []
    while b < n:
        result.append(b)
        a, b = b, a + b
    return result
