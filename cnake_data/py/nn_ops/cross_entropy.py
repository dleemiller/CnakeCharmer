"""Cross entropy loss computation.

Cross entropy loss for n classes with logits and target=0.

Keywords: cross_entropy, loss, neural network, tensor, f32, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def cross_entropy(n: int) -> float:
    """Compute cross entropy loss for n classes, return loss value.

    Args:
        n: Number of classes.

    Returns:
        Cross entropy loss value.
    """
    target = 0

    # Generate logits
    logits = [(i * 17 + 5) % 100 / 10.0 - 5.0 for i in range(n)]

    # Stable softmax: subtract max
    max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute log-sum-exp
    sum_exp = 0.0
    for i in range(n):
        sum_exp += math.exp(logits[i] - max_logit)

    log_sum_exp = math.log(sum_exp) + max_logit

    # Cross entropy: -logits[target] + log_sum_exp
    loss = -logits[target] + log_sum_exp

    return loss
