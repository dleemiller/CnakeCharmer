# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cross entropy loss on f32 tensor (basic Cython, scalar loop).

Keywords: cross_entropy, loss, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp, log
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def cross_entropy(int n):
    """Compute cross entropy loss for n classes, return loss."""
    cdef float *logits = <float *>malloc(n * sizeof(float))
    if not logits:
        raise MemoryError()

    cdef int i
    cdef int target = 0
    cdef float max_logit
    cdef double sum_exp = 0.0
    cdef double loss

    # Generate logits
    for i in range(n):
        logits[i] = (i * 17 + 5) % 100 / 10.0 - 5.0

    # Find max for numerical stability
    max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute log-sum-exp
    for i in range(n):
        sum_exp += exp(<double>(logits[i] - max_logit))

    loss = -<double>logits[target] + log(sum_exp) + <double>max_logit

    free(logits)
    return loss
