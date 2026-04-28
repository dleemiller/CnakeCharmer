import math


def lowest_values(
    n,
    sampler,
    predict,
    simulations,
    best_metric,
    rounds=100,
):
    """Find up to n lowest predicted candidates from sampled points.

    Args:
        n: number of candidate points to keep.
        sampler: callable returning iterable of candidate vectors.
        predict: callable mapping vector -> score.
        simulations: object exposing ``is_new(x)``.
        best_metric: current best score threshold.
        rounds: number of sampling rounds.
    """
    candidate_xs = [None] * n
    candidate_ys = [math.inf] * n
    nth = float(best_metric)
    negative_value = 0

    for _ in range(rounds):
        for x in sampler():
            if not simulations.is_new(x):
                continue

            y = float(predict(x))
            if y < 0.0:
                negative_value += 1
                continue

            if y <= nth:
                i = max(range(n), key=lambda idx: candidate_ys[idx])
                candidate_ys[i] = y
                candidate_xs[i] = x
                nth = min(max(candidate_ys), best_metric)

    filtered = [
        (x, y)
        for x, y in zip(candidate_xs, candidate_ys, strict=False)
        if x is not None and math.isfinite(y)
    ]
    xs = [x for x, _ in filtered]
    ys = [y for _, y in filtered]
    return xs, ys, negative_value
