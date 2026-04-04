"""Compute Doppler-shifted frequencies for approaching sources.

Keywords: physics, doppler, frequency, sound, wave, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def doppler_shift(n: int) -> float:
    """Compute Doppler-shifted frequencies for n approaching sources.

    f0 = 1000 Hz, v_source[i] = (i*7 + 3) % 340 m/s.
    f = f0 * c / (c - v), c = 343 m/s. Returns sum of shifted frequencies.

    Args:
        n: Number of sources.

    Returns:
        Sum of Doppler-shifted frequencies as a float.
    """
    f0 = 1000.0
    c = 343.0

    total = 0.0
    for i in range(n):
        v_source = (i * 7 + 3) % 340
        f = f0 * c / (c - v_source)
        total += f

    return total
