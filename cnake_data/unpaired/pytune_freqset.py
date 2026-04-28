import numpy as np


def vals_to_freqs(freqs, sample_rate):
    return [f * sample_rate for f in freqs]


def filter_duplicate_freqs(freqweights):
    newfreqweights = []
    prev_freq = -1
    prev_weight = -1

    for freq, weight in freqweights:
        new_weight = weight
        if freq == prev_freq:
            new_weight = weight + prev_weight
            newfreqweights.pop()
        newfreqweights.append((freq, new_weight))
        prev_freq = freq
        prev_weight = weight

    return newfreqweights


def from_wave(wave):
    weight = np.abs(np.fft.fft(wave.data))
    freqs = np.abs(np.fft.fftfreq(len(wave.data)))
    freqsl = vals_to_freqs(freqs, wave.sampleRate)

    freqweights = list(zip(freqsl, weight, strict=False))
    freqweights = sorted(freqweights, key=lambda x: x[0])
    if freqweights:
        freqweights.pop(0)
    freqweights = filter_duplicate_freqs(freqweights)
    return freqweights
