"""
Weighted L2 distance between two deterministic signatures.

Sourced from SFT DuckDB blob: 1949531910e5e150b1398a5d9580c37bb6e917c9
Keywords: signature, euclidean distance, l2, weighted norm, numerical
"""

from math import sqrt

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000, 11, 0.35))
def signature_l2(n: int, stride: int, bias: float) -> tuple:
    """Compute weighted L2 distance between two deterministic float signatures."""
    sse = 0.0
    abs_sum = 0.0
    max_abs = 0.0

    for i in range(n):
        a = ((i * stride + 3) % 1000) * 0.01 + bias
        b = ((i * (stride + 4) + 17) % 1000) * 0.01 - bias * 0.5
        diff = a - b
        ad = abs(diff)
        sse += diff * diff
        abs_sum += ad
        if ad > max_abs:
            max_abs = ad

    dist = sqrt(sse)
    mean_abs = abs_sum / n if n > 0 else 0.0
    return (round(dist, 10), round(mean_abs, 10), round(max_abs, 10))
