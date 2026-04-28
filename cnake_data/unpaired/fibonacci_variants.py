_MEMO = [-1] * 1000


def fibonacci_recursive(n):
    """Naive recursive Fibonacci."""
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    """Bottom-up dynamic programming Fibonacci."""
    if n < 2:
        return n
    arr = [0] * (n + 1)
    arr[1] = 1
    for i in range(2, n + 1):
        arr[i] = arr[i - 1] + arr[i - 2]
    return arr[n]


def fibonacci_memoized(n):
    """Top-down memoized Fibonacci using a module-level cache."""
    if n < 2:
        return n
    if _MEMO[n - 1] == -1:
        _MEMO[n - 1] = fibonacci_memoized(n - 1)
    if _MEMO[n - 2] == -1:
        _MEMO[n - 2] = fibonacci_memoized(n - 2)
    _MEMO[n] = _MEMO[n - 1] + _MEMO[n - 2]
    return _MEMO[n]
