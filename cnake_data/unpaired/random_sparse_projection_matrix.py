import random


def sporf_ternary_indices(p, d, k, rng=None):
    """Sample nonzero ternary projection entries using flat index mapping."""
    if rng is None:
        rng = random

    m = int(p * d * k)
    n = p * d
    chosen = rng.sample(range(n), m)

    fi = [0 for _ in range(m)]
    fj = [0 for _ in range(m)]
    w = [rng.choice((-1, 1)) for _ in range(m)]

    for i, v in enumerate(chosen):
        fi[i] = v // d
        fj[i] = v % d

    return fi, fj, w


def sporf_ternary_columns_indices(p, d, lam, rng=None):
    """Column-wise sparse ternary sampling with at least one nnz per column."""
    if rng is None:
        rng = random

    fi = []
    fj = []
    w = []

    base = int(lam)
    for col in range(d):
        # Simple integer-valued surrogate for Poisson(lam) + 1
        nnz = max(1, base + (1 if rng.random() < (lam - base) else 0))
        nnz = min(nnz, p)
        rows = rng.sample(range(p), nnz)
        for r in rows:
            fi.append(r)
            fj.append(col)
            w.append(rng.choice((-1, 1)))

    return fi, fj, w
