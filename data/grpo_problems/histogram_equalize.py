def histogram_equalize(n):
    """Apply histogram equalization to deterministic grayscale pixel data.

    Returns (mean output value, count of pixels at intensity 128, checksum).
    """
    pixels = [0] * n
    for i in range(n):
        v = ((i * 73 + 17) % 256 * (i * 31 + 3) % 256) % 256
        pixels[i] = v

    hist = [0] * 256
    for p in pixels:
        hist[p] += 1

    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    denom = n - cdf_min
    if denom <= 0:
        denom = 1
    transfer = [0] * 256
    for i in range(256):
        if cdf[i] > 0:
            transfer[i] = round((cdf[i] - cdf_min) * 255.0 / denom)

    total = 0
    count_128 = 0
    checksum = 0
    for i in range(n):
        out = transfer[pixels[i]]
        total += out
        if out == 128:
            count_128 += 1
        checksum = (checksum + out * (i & 0xFF)) & 0xFFFFFFFF

    mean_val = round(total / n, 2) if n > 0 else 0.0

    return (mean_val, count_128, checksum)
