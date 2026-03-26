"""Hilbert transform envelope detection on an amplitude-modulated signal.

Uses a naive DFT-based approach to compute the analytic signal magnitude
and returns the sum of the envelope.

Keywords: dsp, envelope, Hilbert, transform, analytic, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def envelope_detection(n: int) -> float:
    """Compute envelope via naive Hilbert transform and return sum.

    Signal: s[i] = sin(i*0.1) * (1 + 0.5*cos(i*0.001)).

    Args:
        n: Signal length.

    Returns:
        Sum of the envelope (analytic signal magnitude).
    """
    two_pi_over_n = 2.0 * math.pi / n

    # Generate signal
    s = [math.sin(i * 0.1) * (1.0 + 0.5 * math.cos(i * 0.001)) for i in range(n)]

    # Compute Hilbert transform via DFT:
    # For each sample, compute the imaginary part of the analytic signal
    # h[i] = sum_k ( s[k] * (-2/n) * sum over positive freq contributions )
    # Simplified: direct quadrature computation using DFT pairs
    total = 0.0
    for i in range(n):
        # Compute quadrature component (Hilbert transform at point i)
        quad = 0.0
        for k in range(n):
            if k == 0 or k == n:
                continue
            # Hilbert transform kernel in frequency domain
            # For odd k contribution
            if k < n // 2:
                quad += s[k] * math.sin(two_pi_over_n * k * i) * 2.0 / n
            elif k > n // 2:
                quad -= s[k] * math.sin(two_pi_over_n * k * i) * 2.0 / n

        envelope = math.sqrt(s[i] * s[i] + quad * quad)
        total += envelope

    return total
