import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRTPIHALF = math.sqrt(math.pi * 0.5)


def gauss_weighted_avg(a, b, amp, stddev, mean):
    """Weighted average of a Gaussian between [a, b]."""
    erf_term = math.erf((mean - a) / (SQRT2 * stddev)) - math.erf((mean - b) / (SQRT2 * stddev))
    weight_avg = (amp * stddev / (b - a)) * erf_term * SQRTPIHALF
    return weight_avg


def gauss_model_discrete(vels, amp, mean, stddev):
    half_chan_width = abs(vels[1] - vels[0]) / 2.0
    vals = np.zeros_like(vels, dtype=float)
    for i, vel in enumerate(vels):
        vals[i] = gauss_weighted_avg(
            vel - half_chan_width, vel + half_chan_width, amp, stddev, mean
        )
    return vals


def sample_at_channels(vels, upsamp_vels, values):
    """Resample values to channel centers by local weighted average bins."""
    spec = np.zeros_like(vels, dtype=float)
    half_chan_width = (vels[1] - vels[0]) / 2.0

    for i, vel in enumerate(vels):
        if half_chan_width < 0:
            bin_mask = np.logical_and(
                upsamp_vels >= vel + half_chan_width, upsamp_vels <= vel - half_chan_width
            )
        else:
            bin_mask = np.logical_and(
                upsamp_vels >= vel - half_chan_width, upsamp_vels <= vel + half_chan_width
            )

        if not np.any(bin_mask):
            raise ValueError("No model values found in bin.")

        spec[i] = values[bin_mask].mean()

    return spec
