DISHARMONY_MAP = {
    0: 0.0,
    1: 10.0,
    2: 3.0,
    3: 3.0,
    4: 2.0,
    5: 1.0,
    6: 10.0,
    7: 1.0,
    8: 3.0,
    9: 6.0,
    10: 3.0,
    11: 10.0,
}


def _overlap_with_inertia(start_a, end_a, start_b, end_b, hearing_inertia=1.5):
    return max(0.0, min(end_a + hearing_inertia, end_b + hearing_inertia) - max(start_a, start_b))


def _pair_disharmony(i, j, starts, ends, pitches):
    overlap = _overlap_with_inertia(starts[i], ends[i], starts[j], ends[j])
    if overlap == 0.0:
        return 0.0

    interval = (pitches[i] - pitches[j]) % 12
    if interval < 0:
        interval = -interval

    return DISHARMONY_MAP.get(interval, 0.0) * overlap


def calculate_disharmony(starts, ends, pitches):
    """Total pairwise disharmony score across all note pairs."""
    n = len(starts)
    res = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            res += _pair_disharmony(i, j, starts, ends, pitches)
    return res
