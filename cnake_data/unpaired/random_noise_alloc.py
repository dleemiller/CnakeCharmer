"""Generate Gaussian noise samples in a tight loop."""

from __future__ import annotations

import random


def random_noise(number=1):
    ran = random.normalvariate
    out = [0.0] * number
    for i in range(number):
        out[i] = ran(0.0, 1.0)
    return out
