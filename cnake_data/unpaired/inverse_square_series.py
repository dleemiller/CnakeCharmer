import time


def inverse_square_sum(n, m):
    """Compute sum_{k=n}^{m-1} 1/k^2."""
    result = 0.0
    for k in range(n, m):
        result += 1.0 / float(k * k)
    return result


def benchmark_inverse_square_loop(repeats=5000, n=1, m=10000):
    """Repeat inverse-square sum and return result plus elapsed seconds."""
    start = time.perf_counter()
    result = 0.0
    for _ in range(repeats):
        result = inverse_square_sum(n, m)
    elapsed = time.perf_counter() - start
    return result, elapsed
