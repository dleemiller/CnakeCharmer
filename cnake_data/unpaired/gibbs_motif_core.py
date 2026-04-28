import math
import random


def count_seq(seq):
    bases = {"a": 0, "c": 1, "g": 2, "t": 3}
    freq = {0: 0, 1: 0, 2: 0, 3: 0}
    int_list = [0] * len(seq)
    for i, ch in enumerate(seq):
        idx = bases[ch]
        freq[idx] += 1
        int_list[i] = idx
    return freq, int_list


def propensity(alignment, w, k, background_freq):
    """Compute position-specific propensity matrix (4 x w)."""
    pseudo = math.sqrt(k) * 0.25
    p = [[0.0] * w for _ in range(4)]

    for j in range(w):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for row in alignment:
            base = row[j]
            if base in counts:
                counts[base] += 1
        for b in range(4):
            p[b][j] = ((counts[b] + pseudo) / (k + pseudo * 4.0)) / background_freq[b]

    return p


def entropy(alignment, background_freq):
    w = len(alignment[0])
    pseudo = 0.1
    h = 0.0

    for j in range(w):
        count = {0: pseudo, 1: pseudo, 2: pseudo, 3: pseudo}
        for i in range(len(alignment)):
            count[alignment[i][j]] += 1

        total = float(len(alignment))
        for b in range(4):
            q = count[b] / total
            h += q * math.log(q / background_freq[b])

    return h / w


def get_opt_offset(p, t_star, n_star, w):
    """Sample motif offset from propensity-induced posterior."""
    pdf = [0.0] * (n_star - w)

    denom = 0.0
    for i in range(n_star - w):
        prod = 1.0
        for j in range(w):
            prod *= p[t_star[i + j]][j]
        denom += prod

    for o in range(n_star - w):
        prod = 1.0
        for j in range(w):
            prod *= p[t_star[o + j]][j]
        pdf[o] = prod / denom if denom else 0.0

    cdf = []
    run = 0.0
    for v in pdf:
        run += v
        cdf.append(run)
    if cdf:
        cdf[-1] = 1.0

    r = random.random()
    for i, v in enumerate(cdf):
        if r <= v:
            return i
    return max(len(cdf) - 1, 0)
