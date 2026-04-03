"""Jensen-Shannon divergence between probability distributions.

Keywords: kl divergence, js divergence, information theory, entropy, probability
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def js_divergence(n):
    """Compute pairwise JS divergence for n probability distributions.

    Args:
        n: Number of distributions (each of dimension 50).

    Returns:
        Tuple of (total_divergence, max_divergence, count_above_threshold).
    """
    d = 50

    # Generate n deterministic probability distributions
    dists = []
    for i in range(n):
        row = []
        total = 0.0
        for j in range(d):
            val = ((i * 7 + j * 13 + 3) % 97) + 1.0
            row.append(val)
            total += val
        # Normalize to sum to 1
        for j in range(d):
            row[j] /= total
        dists.append(row)

    total_div = 0.0
    max_div = 0.0
    count_above = 0
    threshold = 0.1

    for i in range(n):
        for j in range(i + 1, n):
            # Compute JS divergence = 0.5 * (KL(p||m) + KL(q||m))
            # where m = 0.5 * (p + q)
            js = 0.0
            for k in range(d):
                p = dists[i][k]
                q = dists[j][k]
                m = 0.5 * (p + q)
                if p > 1e-300:
                    js += p * math.log(p / m)
                if q > 1e-300:
                    js += q * math.log(q / m)
            js *= 0.5

            total_div += js
            if js > max_div:
                max_div = js
            if js > threshold:
                count_above += 1

    return (total_div, max_div, count_above)
