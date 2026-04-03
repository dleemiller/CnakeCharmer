"""Time-lagged conditional probability estimation.

Computes P(A|B) where event A follows event B by a fixed time delta,
using integer array event matching over deterministic sequences.

Keywords: probability, conditional, time series, lag, statistics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def time_lagged_probability(n: int) -> tuple:
    """Estimate time-lagged conditional probabilities over n timesteps.

    At each timestep t, event A occurs if (t*7 + 3) % 5 == 0,
    and event B occurs if (t*11 + 7) % 7 == 0.
    P(A at t | B at t-delta) is estimated for delta=3.

    We compute:
      count_b: number of timesteps where B occurs
      count_a: number of timesteps where A occurs
      count_ab: number of timesteps t >= delta where B at t-delta AND A at t
      count_ba: number of timesteps t >= delta where A at t-delta AND B at t
      p_ab = count_ab / count_b (prob of A given prior B)
      p_ba = count_ba / count_a (prob of B given prior A)

    Args:
        n: Number of timesteps.

    Returns:
        Tuple of (p_ab, p_ba).
    """
    delta = 3

    # Pre-compute event arrays as 0/1 integers
    event_a = [0] * n
    event_b = [0] * n
    for t in range(n):
        if (t * 7 + 3) % 5 == 0:
            event_a[t] = 1
        if (t * 11 + 7) % 7 == 0:
            event_b[t] = 1

    # Count marginals
    count_a = 0
    count_b = 0
    for t in range(n):
        count_a += event_a[t]
        count_b += event_b[t]

    # Count joint occurrences with lag
    count_ab = 0  # B at t-delta, A at t
    count_ba = 0  # A at t-delta, B at t
    for t in range(delta, n):
        if event_b[t - delta] == 1 and event_a[t] == 1:
            count_ab += 1
        if event_a[t - delta] == 1 and event_b[t] == 1:
            count_ba += 1

    p_ab = count_ab / max(count_b, 1)
    p_ba = count_ba / max(count_a, 1)

    return (p_ab, p_ba)
