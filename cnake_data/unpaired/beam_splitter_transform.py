from __future__ import annotations


def two_bs2x4_transform(
    state_in: list[list[complex]], t1: float, r1: float, t2: float, r2: float
) -> list[list[list[list[complex]]]]:
    sz = len(state_in)
    out = [[[[0j for _ in range(sz)] for _ in range(sz)] for _ in range(sz)] for _ in range(sz)]
    for m in range(sz):
        for n in range(sz):
            base = state_in[m][n]
            for k in range(m + 1):
                c1 = (t1 ** (m - k)) * ((1j * r1) ** k)
                for ell in range(n + 1):
                    c2 = (t2 ** (n - ell)) * ((1j * r2) ** ell)
                    out[k][m - k][ell][n - ell] += base * c1 * c2
    return out
