"""1D median filter with window size 5 on a NumPy array.

Applies a sliding window median filter. Python version uses a
sorted window approach.

Keywords: dsp, median, filter, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def numpy_median_filter(n: int) -> float:
    """Apply 1D median filter (window=5) and return sum.

    Args:
        n: Length of the input array.

    Returns:
        Sum of the filtered output.
    """
    rng = np.random.RandomState(42)
    data = rng.standard_normal(n)
    out = np.empty(n, dtype=np.float64)
    half = 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = sorted(data[lo:hi])
        out[i] = window[len(window) // 2]
    return float(np.sum(out))
