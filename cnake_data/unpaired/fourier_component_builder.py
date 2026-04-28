import math


def wn(n, freq):
    """Angular frequency for harmonic n."""
    return (2.0 * math.pi * n) * freq


def wn_array(nterms, freq):
    """Angular frequencies for harmonics 1..len(nterms)."""
    return [wn(i + 1, freq) for i in range(len(nterms))]


def to_radians(degrees):
    return degrees * (math.pi / 180.0)


def fourier_sum(x, coefficients, freq, phase, nterms):
    """Partial Fourier series sum following trace semantics."""
    ws = wn_array(nterms, freq)
    phase_rad = to_radians(phase)

    total = 0.0
    for c, w, n in zip(coefficients, ws, nterms, strict=False):
        total += c * math.sin(w * x + n * phase_rad)
    return total


def upsample_component(amp, phase, duration, times, coefficients, nterms):
    """Evaluate scaled Fourier partial sum at each time sample."""
    frequency = 1.0 / duration
    out = [0.0 for _ in range(len(times))]
    for i, t in enumerate(times):
        out[i] = amp * fourier_sum(t, coefficients, frequency, phase, nterms)
    return out


def single_component(freq, duration, upsampled, samples):
    """Sample one periodic component by index wrapping."""
    cycles = freq * duration
    idx = [int(round(i * cycles)) % samples for i in range(samples)]
    return [upsampled[i] for i in idx]
