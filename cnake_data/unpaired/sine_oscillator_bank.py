from __future__ import annotations

import math


def synth_sine_bank(
    freqs: list[float],
    amps: list[float],
    phases: list[float],
    sample_rate: float,
    n_samples: int,
) -> list[float]:
    if not (len(freqs) == len(amps) == len(phases)):
        raise ValueError("freqs/amps/phases length mismatch")

    out = [0.0] * n_samples
    twopi = 2.0 * math.pi
    for t in range(n_samples):
        tt = t / sample_rate
        s = 0.0
        for i in range(len(freqs)):
            s += amps[i] * math.sin(twopi * freqs[i] * tt + phases[i])
        out[t] = s
    return out
