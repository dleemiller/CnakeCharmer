# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute analytic signal envelope via discrete Hilbert transform convolution (Cython-optimized).

Uses the truncated Hilbert kernel h[k] = 2/(pi*k) for odd k, 0 for even k,
convolved directly with the signal to produce the quadrature component.

Keywords: dsp, hilbert, envelope, analytic signal, convolution, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(8000,))
def hilbert_envelope(int n):
    """Compute envelope of analytic signal via discrete Hilbert transform."""
    cdef int i, k, j, m, kernel_len, quarter_idx
    cdef double pi2 = 2.0 * M_PI
    cdef double inv_pi = 1.0 / M_PI
    cdef double acc, env, env_sum = 0.0, env_max = 0.0, env_at_quarter = 0.0

    cdef double *signal = <double *>malloc(n * sizeof(double))
    if not signal:
        raise MemoryError()

    # Generate signal
    for i in range(n):
        signal[i] = sin(pi2 * i * 5.0 / n) + 0.3 * cos(pi2 * i * 13.0 / n)

    # Build truncated Hilbert kernel
    m = n // 2
    if m > 64:
        m = 64
    kernel_len = 2 * m + 1
    cdef double *kernel = <double *>malloc(kernel_len * sizeof(double))
    cdef double *quadrature = <double *>malloc(n * sizeof(double))
    if not kernel or not quadrature:
        free(signal)
        if kernel:
            free(kernel)
        if quadrature:
            free(quadrature)
        raise MemoryError()

    for k in range(kernel_len):
        kernel[k] = 0.0
    for k in range(-m, m + 1):
        if k % 2 != 0:
            kernel[k + m] = 2.0 * inv_pi / k

    # Convolve to get quadrature component
    for i in range(n):
        acc = 0.0
        for k in range(-m, m + 1):
            j = i - k
            if 0 <= j < n:
                acc += signal[j] * kernel[k + m]
        quadrature[i] = acc

    # Compute envelope
    quarter_idx = n // 4
    for i in range(n):
        env = sqrt(signal[i] * signal[i] + quadrature[i] * quadrature[i])
        env_sum += env
        if env > env_max:
            env_max = env
        if i == quarter_idx:
            env_at_quarter = env

    free(signal)
    free(kernel)
    free(quadrature)
    return (env_sum, env_max, env_at_quarter)
