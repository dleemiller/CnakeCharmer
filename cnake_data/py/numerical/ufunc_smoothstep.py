"""Hermite smoothstep interpolation using np.vectorize.

Python-level per-element smoothstep as baseline for a Cython ufunc.

Keywords: numerical, smoothstep, interpolation, hermite, numpy, ufunc, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


def _smoothstep(x):
    t = (x - 0.2) / (0.8 - 0.2)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


_smoothstep_vec = np.vectorize(_smoothstep)


@python_benchmark(args=(1000000,))
def ufunc_smoothstep(n: int) -> float:
    """Apply smoothstep to n uniform-random values and return sum.

    edge0=0.2, edge1=0.8; t = clamp((x-edge0)/(edge1-edge0), 0, 1)
    result = t*t*(3 - 2*t)

    Args:
        n: Number of elements.

    Returns:
        Sum of smoothstep outputs.
    """
    rng = np.random.RandomState(42)
    arr = rng.random(n)
    result = _smoothstep_vec(arr)
    return float(np.sum(result))
