"""Evaluate cubic polynomials over deterministic inputs.

Keywords: numerical, polynomial, horner, modular arithmetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _poly_eval(v: int, mod: int) -> int:
    coeffs = (3, -2, 5, -7)
    acc = coeffs[0]
    for c in coeffs[1:]:
        acc = acc * v + c
    return acc % mod


@python_benchmark(args=(200000,))
def polynomial_horner_triplet(n: int) -> tuple:
    """Evaluate a cubic with Horner's rule and return summary statistics."""
    mod = 1_000_003
    total = 0
    alt_xor = 0
    max_val = 0

    for i in range(n):
        v = ((i * 37 + 11) % 1009) - 504
        p = _poly_eval(v, mod)
        total = (total + p) % mod
        alt_xor = (alt_xor ^ ((p + i) & 0xFFFFFFFF)) & 0xFFFFFFFF
        if p > max_val:
            max_val = p

    return (total, alt_xor, max_val)
