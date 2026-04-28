"""Mean reciprocal rank over grouped predictions."""

from __future__ import annotations


def mrr(l, p, g, n_groups: int):
    index = 0
    score = 0.0
    for gr in range(n_groups):
        ggr = g[gr]
        idx_one = index
        for i in range(ggr):
            if l[index + i] == 1:
                idx_one = i + index
        our_guess = p[idx_one]
        times_worse = 0
        for i in range(ggr):
            if p[index + i] >= our_guess:
                times_worse += 1
        rr = 1.0 / times_worse
        time = gr + 1.0
        score = ((time - 1) / time) * score + (1 / time) * rr
        index += ggr
    return score
