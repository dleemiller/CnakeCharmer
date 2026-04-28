import itertools


def subseq(seq):
    """Generate non-empty sorted subsequences joined by commas."""
    n = len(seq)
    for i in range(1, n + 1):
        for s in itertools.combinations(seq, i):
            yield ",".join(sorted(s))


def v_function(combo, conversions):
    """Sum conversion values across all non-empty sub-combinations."""
    worth = 0.0
    for subset in subseq(combo.split(",")):
        worth += conversions.get(subset, 0.0)
    return worth


def get_v_values(writers, conversions):
    """Compute v-values for all non-empty combinations of writers."""
    out = {}
    all_subseq = list(subseq(list(writers)))
    for combo in all_subseq:
        out[combo] = v_function(combo, conversions)
    return out
