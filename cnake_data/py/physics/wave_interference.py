"""Compute interference pattern of point sources.

Keywords: physics, wave, interference, diffraction, amplitude, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def wave_interference(n: int) -> float:
    """Compute interference pattern of n point sources at 1000 observation points.

    Source[i] at x=i*0.5, wavelength=1.0.
    Amplitude at obs point = sum(sin(2*pi*r/lambda)/r).
    Returns sum of intensities (amplitude^2) at all observation points.

    Args:
        n: Number of point sources.

    Returns:
        Sum of intensities as a float.
    """
    wavelength = 1.0
    two_pi_over_lam = 2.0 * math.pi / wavelength
    n_obs = 1000

    total_intensity = 0.0
    for obs in range(n_obs):
        obs_x = obs * 0.01 - 5.0
        obs_y = 10.0
        amplitude = 0.0
        for i in range(n):
            src_x = i * 0.5
            dx = obs_x - src_x
            r = math.sqrt(dx * dx + obs_y * obs_y)
            if r > 0:
                amplitude += math.sin(two_pi_over_lam * r) / r
        total_intensity += amplitude * amplitude

    return total_intensity
