# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Numerically stable softmax on groups.

Keywords: softmax, stable, neural network, activation, exp, cython
"""

from libc.math cimport exp
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def softmax_stable(int n):
    """Apply numerically stable softmax on groups of 100 and return sum."""
    cdef int group_size = 100
    cdef int num_groups = n // group_size
    cdef double total = 0.0
    cdef int g, i, offset
    cdef double max_val, v, exp_sum

    cdef double *vals = <double *>malloc(group_size * sizeof(double))
    cdef double *exp_vals = <double *>malloc(group_size * sizeof(double))
    if not vals or not exp_vals:
        if vals: free(vals)
        if exp_vals: free(exp_vals)
        raise MemoryError()

    for g in range(num_groups):
        offset = g * group_size
        # Compute values and find max
        max_val = (((offset) * 17 + 5) % 1000) / 100.0
        vals[0] = max_val
        for i in range(1, group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 100.0
            vals[i] = v
            if v > max_val:
                max_val = v
        # Compute exp and sum
        exp_sum = 0.0
        for i in range(group_size):
            exp_vals[i] = exp(vals[i] - max_val)
            exp_sum += exp_vals[i]
        # Sum softmax outputs
        for i in range(group_size):
            total += exp_vals[i] / exp_sum

    free(vals)
    free(exp_vals)
    return total
