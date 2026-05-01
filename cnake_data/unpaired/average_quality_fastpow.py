from __future__ import annotations

import math


def compute_average_quality(phred_list: list[float], length: int) -> float:
    prob = 0.0
    for q in phred_list:
        prob += 10.0 ** (q / -10.0)
    return -10.0 * math.log10(prob / length)
