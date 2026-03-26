"""
Chinese Remainder Theorem — solve and sum congruence systems.

Keywords: math, chinese remainder theorem, CRT, extended GCD, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def chinese_remainder(n: int) -> int:
    """Solve n pairwise CRT systems and sum the solutions mod 10^9+7.

    For consecutive pairs (i, i+1): a1 = (i*7+3) % 100, m1 = (i*3+11),
    a2 = ((i+1)*7+3) % 100, m2 = ((i+1)*3+11).
    Uses extended GCD to combine each pair.
    Sum all solutions mod 10^9+7.

    Args:
        n: Number of pairwise systems (uses n+1 congruences).

    Returns:
        Sum of all pairwise CRT solutions mod 10^9+7.
    """
    MOD = 1000000007

    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = extended_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1

    total = 0

    for i in range(n):
        a1 = (i * 7 + 3) % 100
        m1 = i * 3 + 11
        a2 = ((i + 1) * 7 + 3) % 100
        m2 = (i + 1) * 3 + 11

        g, p, q = extended_gcd(m1, m2)
        lcm = m1 * (m2 // g)
        diff = a2 - a1
        if diff % g != 0:
            # No solution for this pair, skip
            continue
        solution = (a1 + m1 * (diff // g % (m2 // g)) * p) % lcm
        if solution < 0:
            solution += lcm
        total = (total + solution) % MOD

    return total
