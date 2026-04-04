"""Sliding window statistics (min, max, mean, variance) over a sequence.

Keywords: numerical, sliding window, statistics, running stats, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def running_stats(n: int) -> tuple:
    """Compute sliding window stats (w=100) over a deterministic sequence.

    Sequence: v[i] = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0
    Window size w=100. For each position i from w-1 to n-1, maintain a sliding
    window sum and sum_sq (add new element, drop old element). For min/max,
    scan the window each step.

    Returns stats at the last window position i=n-1:
        (int(mean * 1e9), int(variance * 1e9), int(min * 1e9), int(max * 1e9))
    """
    w = 100
    # Generate the sequence
    v = [0.0] * n
    for i in range(n):
        v[i] = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0

    # Prime the first window
    window_sum = 0.0
    window_sum_sq = 0.0
    for i in range(w):
        window_sum += v[i]
        window_sum_sq += v[i] * v[i]

    last_mean = 0.0
    last_variance = 0.0
    last_min = 0.0
    last_max = 0.0

    for i in range(w - 1, n):
        if i == w - 1:
            # First window already primed
            pass
        else:
            # Slide: add new element, drop old element
            window_sum += v[i] - v[i - w]
            window_sum_sq += v[i] * v[i] - v[i - w] * v[i - w]

        mean = window_sum / w
        variance = window_sum_sq / w - mean * mean

        # Scan window for min/max
        win_min = v[i - w + 1]
        win_max = v[i - w + 1]
        for j in range(i - w + 2, i + 1):
            if v[j] < win_min:
                win_min = v[j]
            if v[j] > win_max:
                win_max = v[j]

        last_mean = mean
        last_variance = variance
        last_min = win_min
        last_max = win_max

    return (
        int(last_mean * 1e9),
        int(last_variance * 1e9),
        int(last_min * 1e9),
        int(last_max * 1e9),
    )
