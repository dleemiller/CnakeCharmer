"""Batch evaluation of neural network activation functions.

Keywords: activation function, sigmoid, relu, tanh, neural network, batch
"""

import math

from cnake_data.benchmarks import python_benchmark


def _sigmoid(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def _tanh_act(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def _softplus(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1.0 + math.exp(z))


def _elu(z):
    return z if z > 0.0 else math.exp(z) - 1.0


def _selu(z):
    lam = 1.0507009873554805
    alpha = 1.6732632423543773
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1.0)


@python_benchmark(args=(3000,))
def activation_functions(n):
    """Apply multiple activation functions to n input values and accumulate.

    Args:
        n: Number of input values.

    Returns:
        Tuple of (sigmoid_sum, tanh_sum, softplus_sum, elu_sum, selu_sum).
    """
    sig_sum = 0.0
    tanh_sum = 0.0
    sp_sum = 0.0
    elu_sum = 0.0
    selu_sum = 0.0

    for i in range(n):
        z = (i - n / 2.0) / (n / 10.0)
        sig_sum += _sigmoid(z)
        tanh_sum += _tanh_act(z)
        sp_sum += _softplus(z)
        elu_sum += _elu(z)
        selu_sum += _selu(z)

    return (sig_sum, tanh_sum, sp_sum, elu_sum, selu_sum)
