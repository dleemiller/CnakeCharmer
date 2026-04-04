"""Compute analytic signal envelope via discrete Hilbert transform convolution.

Uses the truncated Hilbert kernel h[k] = 2/(pi*k) for odd k, 0 for even k,
convolved directly with the signal to produce the quadrature component.

Keywords: dsp, hilbert, envelope, analytic signal, convolution, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(8000,))
def hilbert_envelope(n: int) -> tuple:
    """Compute envelope of analytic signal via discrete Hilbert transform.

    Signal: s[i] = sin(2*pi*i*5/n) + 0.3*cos(2*pi*i*13/n)

    Hilbert transform via direct convolution with truncated kernel:
        h[k] = 2/(pi*k) for odd k, 0 for even k

    Kernel is truncated to length 2*M+1 with M = min(n//2, 64).

    Args:
        n: Signal length.

    Returns:
        Tuple of (envelope_sum, envelope_max, envelope_at_quarter).
    """
    pi2 = 2.0 * math.pi
    inv_pi = 1.0 / math.pi

    # Generate signal
    signal = [0.0] * n
    for i in range(n):
        signal[i] = math.sin(pi2 * i * 5.0 / n) + 0.3 * math.cos(pi2 * i * 13.0 / n)

    # Build truncated Hilbert kernel
    m = min(n // 2, 64)
    kernel_len = 2 * m + 1
    kernel = [0.0] * kernel_len
    for k in range(-m, m + 1):
        if k % 2 != 0:
            kernel[k + m] = 2.0 * inv_pi / k

    # Convolve to get quadrature component
    quadrature = [0.0] * n
    for i in range(n):
        acc = 0.0
        for k in range(-m, m + 1):
            j = i - k
            if 0 <= j < n:
                acc += signal[j] * kernel[k + m]
        quadrature[i] = acc

    # Compute envelope
    env_sum = 0.0
    env_max = 0.0
    quarter_idx = n // 4
    env_at_quarter = 0.0
    for i in range(n):
        env = math.sqrt(signal[i] * signal[i] + quadrature[i] * quadrature[i])
        env_sum += env
        if env > env_max:
            env_max = env
        if i == quarter_idx:
            env_at_quarter = env

    return (env_sum, env_max, env_at_quarter)
