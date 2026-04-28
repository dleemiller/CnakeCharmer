import math
from collections import defaultdict


def calculate_cooccurrence_weights(user_log_times, rating_times_map):
    """Build symmetric pairwise co-occurrence weights.

    Args:
        user_log_times: iterable of ``(user_log, user_time)`` where ``user_log``
            is an ordered list of item ids.
        rating_times_map: mapping item_id -> frequency-like positive float.

    Returns:
        Nested dict: ``weights[item_i][item_j] = score``.
    """
    weights = defaultdict(dict)

    for user_log, user_time in user_log_times:
        scale = math.log1p(user_time)
        if scale == 0.0:
            continue

        n = len(user_log)
        for idx in range(n):
            i1 = user_log[idx]
            rt1 = rating_times_map[i1]
            for j in range(idx + 1, n):
                i2 = user_log[j]
                rt2 = rating_times_map[i2]
                delta = 1.0 / (scale * rt1 * rt2)

                weights[i1][i2] = weights[i1].get(i2, 0.0) + delta
                weights[i2][i1] = weights[i2].get(i1, 0.0) + delta

    return {k: dict(v) for k, v in weights.items()}
