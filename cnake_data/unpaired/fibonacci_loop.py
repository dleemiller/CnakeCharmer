def fib_loop(n):
    """Iterative Fibonacci using local integer state."""
    a, b = 0, 1
    for _ in range(n):
        a, b = a + b, a
    return a
