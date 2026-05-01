"""Monte Carlo estimate for probability of rolling enough sixes."""

from __future__ import annotations

import random


def plain_cython(n_trials, n_dice, n_sixes):
    total = 0
    for _ in range(n_trials):
        count = 0
        for _ in range(n_dice):
            roll = random.randint(1, 6)
            if roll == 6:
                count += 1
        if count >= n_sixes:
            total += 1
    return float(total) / n_trials
