"""
Estimate conditional event probability with a lag window.

Sourced from SFT DuckDB blob: 109d9d93806ed7cd954b046b09e9ce7bab68c8b1
Keywords: conditional probability, event stream, lagged dependency, algorithms
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(6, 100000, 2))
def conditional_event_prob(window: int, rows: int, delta: int) -> tuple:
    """Compute P(B->A), P(A), P(B) on deterministic set-valued event rows."""
    if delta < 0:
        raise ValueError("delta must be non-negative")

    events = []
    for i in range(rows):
        row = set()
        for j in range(window):
            if ((i + 3) * (j + 5) + 11) % 7 < 3:
                row.add(j)
        events.append(row)

    a_event = max(0, window - 1)
    b_event = 0
    p_a = 0
    p_b = 0
    p_ba = 0

    for i in range(rows):
        if a_event in events[i]:
            p_a += 1
        if b_event in events[i]:
            p_b += 1

    for i in range(delta, rows):
        if b_event in events[i - delta] and a_event in events[i]:
            p_ba += 1

    p_a_f = p_a / rows if rows > 0 else 0.0
    p_b_f = p_b / rows if rows > 0 else 0.0
    denom = max(1, rows - delta)
    p_ba_f = p_ba / denom
    return (round(p_ba_f, 10), round(p_a_f, 10), round(p_b_f, 10))
