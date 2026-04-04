"""Cooley-Tukey radix-2 FFT of n complex values (n must be a power of 2).

Keywords: dsp, fft, cooley-tukey, radix-2, fourier, complex, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(16384,))
def fft_radix2(n: int) -> tuple:
    """Iterative Cooley-Tukey FFT on n complex values (largest power of 2 <= n).

    Input: x[k] = cos(2*pi*k/size) + i*sin(2*pi*k/size)  (single complex tone at bin 1).
    Uses in-place butterfly with twiddle-factor recurrence.

    Args:
        n: Determines transform size: largest power of 2 <= n.

    Returns:
        Tuple of (mag_sum, mag_at_quarter, real_at_eighth) — magnitude sum of all
        output bins, magnitude at bin size//4, real part at bin size//8.
    """
    # Largest power of 2 <= n
    size = 1
    while size * 2 <= n:
        size *= 2

    log2_size = size.bit_length() - 1

    # Input: pure complex tone at frequency 1
    TWO_PI = 2.0 * math.pi
    x_r = [math.cos(TWO_PI * k / size) for k in range(size)]
    x_i = [math.sin(TWO_PI * k / size) for k in range(size)]

    # Bit-reversal permutation
    for i in range(size):
        j = 0
        tmp = i
        for _ in range(log2_size):
            j = (j << 1) | (tmp & 1)
            tmp >>= 1
        if j > i:
            x_r[i], x_r[j] = x_r[j], x_r[i]
            x_i[i], x_i[j] = x_i[j], x_i[i]

    # Butterfly stages
    length = 2
    while length <= size:
        half = length >> 1
        angle = -TWO_PI / length
        wr = math.cos(angle)
        wi = math.sin(angle)
        for start in range(0, size, length):
            cr = 1.0
            ci = 0.0
            for j in range(half):
                ur = x_r[start + j]
                ui = x_i[start + j]
                vr = x_r[start + j + half] * cr - x_i[start + j + half] * ci
                vi = x_r[start + j + half] * ci + x_i[start + j + half] * cr
                x_r[start + j] = ur + vr
                x_i[start + j] = ui + vi
                x_r[start + j + half] = ur - vr
                x_i[start + j + half] = ui - vi
                # Advance twiddle factor
                cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
        length <<= 1

    # Summary statistics
    mag_sum = sum(math.sqrt(x_r[k] * x_r[k] + x_i[k] * x_i[k]) for k in range(size))
    q = size >> 2
    mag_at_quarter = math.sqrt(x_r[q] * x_r[q] + x_i[q] * x_i[q])
    e = size >> 3
    real_at_eighth = x_r[e]

    return (mag_sum, mag_at_quarter, real_at_eighth)
