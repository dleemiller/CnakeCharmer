"""Find strongest short-lag cyclic autocorrelation in an integer signal.

Keywords: numerical, autocorrelation, cyclic, signal, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(8000,))
def cyclic_autocorr_peak(n: int) -> tuple:
    """Compute cyclic autocorrelation for lags 1..16 and return peak info."""
    seq = [0] * n
    for i in range(n):
        seq[i] = ((i * 29 + 17) % 31) - 15

    best_lag = 1
    best_val = -(10**18)
    lag1_val = 0

    for lag in range(1, 17):
        total = 0
        for i in range(n):
            total += seq[i] * seq[(i + lag) % n]
        if lag == 1:
            lag1_val = total
        if total > best_val:
            best_val = total
            best_lag = lag

    return (best_lag, best_val, lag1_val)
