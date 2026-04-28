import math

_bandpass_lookup = [0.0] * 256
_last_sigma = 0.0
_last_gain = 0.0
_last_center = 0


def update_bandpass_lookup(center, sigma, gain):
    """Precompute hue bandpass lookup in [0, 255]."""
    global _last_center, _last_sigma, _last_gain
    _last_center = center
    _last_sigma = sigma
    _last_gain = gain

    mu = 0.5
    for n in range(256):
        h = n / 255.0
        h = h + (0.5 - (center / 255.0))
        if h > 1.0:
            h -= 1.0
        if h < 0.0:
            h += 1.0

        f = math.exp(-((h - mu) ** 2 / (2.0 * sigma**2)))
        f = f * gain
        if f > 1.0:
            f = 1.0
        _bandpass_lookup[n] = f


def hue_mask(image, center, sigma, gain, mask_gain):
    """Apply hue/sat/val mask in-place on HSV-like uint8 image."""
    if center != _last_center or sigma != _last_sigma or gain != _last_gain:
        update_bandpass_lookup(center, sigma, gain)

    height = len(image)
    width = len(image[0]) if height else 0

    for y in range(height):
        for x in range(width):
            ih = image[y][x][0]
            s = image[y][x][1] / 255.0
            v = image[y][x][2] / 255.0

            h = _bandpass_lookup[ih]
            f = (h * s * v) * mask_gain
            if f > 1.0:
                f = 1.0

            mask = int(f * 255)
            image[y][x][0] = mask
            image[y][x][1] = mask
            image[y][x][2] = mask

    return image
