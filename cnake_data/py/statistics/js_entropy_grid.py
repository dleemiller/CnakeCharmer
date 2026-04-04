"""Compute KL/JS-like divergence metrics on generated probability grids.

Keywords: statistics, entropy, kl divergence, js divergence, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(0.03, 0.07, 220, 240, 1e-12))
def js_entropy_grid(p_bias: float, q_bias: float, rows: int, cols: int, eps: float) -> tuple:
    """Generate two distributions and compute symmetric divergence metrics."""
    kl_pq = 0.0
    kl_qp = 0.0

    for i in range(rows):
        row_scale = 1.0 + (i & 7) * 0.03
        for j in range(cols):
            base = ((i * 131 + j * 17 + 29) % 1000) / 1000.0
            p = eps + ((base + p_bias) % 1.0) * row_scale
            q = eps + ((base * 0.73 + q_bias + 0.11) % 1.0) * (2.0 - row_scale * 0.25)
            kl_pq += p * math.log(p / q)
            kl_qp += q * math.log(q / p)

    js = 0.5 * (kl_pq + kl_qp)
    return (kl_pq, kl_qp, js)
