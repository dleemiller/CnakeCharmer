"""Monte-Carlo probability of rolling at least N sixes."""

from __future__ import annotations

import random


def dice6_py(num_trials, num_dice, min_sixes):
    successes = 0
    for _ in range(num_trials):
        sixes = 0
        for _ in range(num_dice):
            if random.randint(1, 6) == 6:
                sixes += 1
        if sixes >= min_sixes:
            successes += 1
    return float(successes) / num_trials
