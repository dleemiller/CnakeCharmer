import math


def gauss_blur_1d(n):
    """Apply 1D Gaussian blur to a deterministic signal.

    Returns (sum of output, max of output, min of output).
    """
    signal = [0.0] * n
    for i in range(n):
        signal[i] = math.sin(i * 0.01) + 0.5 * math.sin(i * 0.03)

    radius = 5
    sigma = 1.5
    kernel_size = 2 * radius + 1
    kernel = [0.0] * kernel_size
    kernel_sum = 0.0
    for i in range(kernel_size):
        x = i - radius
        kernel[i] = math.exp(-0.5 * (x / sigma) ** 2)
        kernel_sum += kernel[i]

    for i in range(kernel_size):
        kernel[i] /= kernel_sum

    output = [0.0] * n
    for i in range(n):
        val = 0.0
        for k in range(kernel_size):
            j = i + k - radius
            if j < 0:
                j = 0
            elif j >= n:
                j = n - 1
            val += signal[j] * kernel[k]
        output[i] = val

    sum_out = 0.0
    max_out = output[0]
    min_out = output[0]
    for v in output:
        sum_out += v
        if v > max_out:
            max_out = v
        if v < min_out:
            min_out = v

    return (round(sum_out, 4), round(max_out, 6), round(min_out, 6))
