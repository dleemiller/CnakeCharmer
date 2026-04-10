def fibonacci_sequence(n):
    """Compute the first n Fibonacci numbers.

    Returns a list of the first n Fibonacci numbers.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i - 1] + result[i - 2])
    return result
