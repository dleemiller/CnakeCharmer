import math


def threshold_exp(values, threshold=0.5):
    """Apply exp(x) when x > threshold, else 0.0."""
    out = [0.0] * len(values)
    for i, value in enumerate(values):
        if value > threshold:
            out[i] = math.exp(value)
    return out


def threshold_exp_many(rows, threshold=0.5):
    """Batch version of ``threshold_exp`` for a list of rows."""
    return [threshold_exp(row, threshold=threshold) for row in rows]
