# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch evaluation of neural network activation functions.

Keywords: activation function, sigmoid, relu, tanh, neural network, batch, cython
"""

from libc.math cimport exp, tanh, log

from cnake_data.benchmarks import cython_benchmark


cdef inline double _sigmoid(double z) nogil:
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + exp(-z))


cdef inline double _tanh_act(double z) nogil:
    z = max(-60.0, min(60.0, 2.5 * z))
    return tanh(z)


cdef inline double _softplus(double z) nogil:
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * log(1.0 + exp(z))


cdef inline double _elu(double z) nogil:
    return z if z > 0.0 else exp(z) - 1.0


cdef inline double _selu(double z) nogil:
    cdef double lam = 1.0507009873554805
    cdef double alpha = 1.6732632423543773
    return lam * z if z > 0.0 else lam * alpha * (exp(z) - 1.0)


@cython_benchmark(syntax="cy", args=(3000,))
def activation_functions(int n):
    """Apply multiple activation functions to n input values and accumulate.

    Args:
        n: Number of input values.

    Returns:
        Tuple of (sigmoid_sum, tanh_sum, softplus_sum, elu_sum, selu_sum).
    """
    cdef double sig_sum = 0.0
    cdef double tanh_sum = 0.0
    cdef double sp_sum = 0.0
    cdef double elu_sum = 0.0
    cdef double selu_sum = 0.0
    cdef double z
    cdef int i
    cdef double half_n = n / 2.0
    cdef double tenth_n = n / 10.0

    for i in range(n):
        z = (i - half_n) / tenth_n
        sig_sum += _sigmoid(z)
        tanh_sum += _tanh_act(z)
        sp_sum += _softplus(z)
        elu_sum += _elu(z)
        selu_sum += _selu(z)

    return (sig_sum, tanh_sum, sp_sum, elu_sum, selu_sum)
