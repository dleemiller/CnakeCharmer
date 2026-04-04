"""Pitch detection using Average Magnitude Difference Function (AMDF).

Keywords: dsp, pitch detection, amdf, signal processing, frequency, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80000,))
def pitch_detect_amdf(n: int) -> tuple:
    """Detect pitch of a deterministic signal using AMDF.

    Generates a signal with known frequency components, computes AMDF
    for a range of lags, and finds the lag with minimum AMDF value.

    Args:
        n: Signal length.

    Returns:
        Tuple of (best_lag, min_amdf_value, amdf_energy) where amdf_energy
        is the sum of squared AMDF values.
    """
    # Generate deterministic signal: sum of two sinusoids
    # f1=100Hz, f2=200Hz at sample_rate=8000
    sample_rate = 8000.0
    f1 = 100.0
    f2 = 200.0
    signal = [0.0] * n
    for i in range(n):
        t = i / sample_rate
        signal[i] = 0.7 * math.sin(2.0 * math.pi * f1 * t) + 0.3 * math.sin(2.0 * math.pi * f2 * t)

    # Compute AMDF for lags 20..400 (corresponding to 20-400 Hz range)
    min_lag = 20
    max_lag = min(401, n // 2)
    num_lags = max_lag - min_lag
    amdf = [0.0] * num_lags

    # Window size for AMDF computation
    window = min(n // 2, 2000)

    for li in range(num_lags):
        lag = min_lag + li
        acc = 0.0
        for i in range(window):
            diff = signal[i] - signal[i + lag]
            if diff < 0.0:
                diff = -diff
            acc += diff
        amdf[li] = acc / window

    # Find minimum AMDF (best lag = detected pitch period)
    best_lag = min_lag
    min_val = amdf[0]
    for li in range(1, num_lags):
        if amdf[li] < min_val:
            min_val = amdf[li]
            best_lag = min_lag + li

    # Compute energy of AMDF curve
    energy = 0.0
    for li in range(num_lags):
        energy += amdf[li] * amdf[li]

    return (best_lag, min_val, energy)
