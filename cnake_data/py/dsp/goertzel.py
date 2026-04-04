"""Goertzel algorithm to detect a specific frequency bin in a signal.

Computes the magnitude, phase cosine, and power at a target frequency using
the Goertzel recurrence.

Keywords: dsp, Goertzel, frequency, detection, DFT, power, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def goertzel(n: int) -> tuple:
    """Compute DFT metrics at target frequency using Goertzel algorithm.

    Signal: s[i] = sin(2*pi*i*100/n) + 0.5*sin(2*pi*i*300/n).
    Target frequency bin: 100.

    Args:
        n: Signal length.

    Returns:
        Tuple of (magnitude, phase_cos, power).
    """
    target_bin = 100
    two_pi = 2.0 * math.pi
    coeff = 2.0 * math.cos(two_pi * target_bin / n)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    for i in range(n):
        sample = math.sin(two_pi * i * 100 / n) + 0.5 * math.sin(two_pi * i * 300 / n)
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    power = s1 * s1 + s2 * s2 - coeff * s1 * s2

    # Real and imaginary parts of DFT at target bin
    w = two_pi * target_bin / n
    real_part = s1 - s2 * math.cos(w)
    imag_part = s2 * math.sin(w)

    magnitude = math.sqrt(real_part * real_part + imag_part * imag_part)
    phase_cos = real_part / magnitude if magnitude > 0.0 else 0.0

    return (magnitude, phase_cos, power)
