"""Process pixel data by color channel using enum-based dispatch.

Keywords: image processing, color, channel, enum, pixel, blend, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

RED = 0
GREEN = 1
BLUE = 2


@python_benchmark(args=(100000,))
def cpdef_enum_color_blend(n: int) -> int:
    """Process n pixels by applying channel-specific transformations.

    Pixel RGB values: r = (i * 41 + 7) % 256, g = (i * 59 + 13) % 256, b = (i * 71 + 3) % 256.
    Transform: RED *= 0.8, GREEN *= 1.1 (cap 255), BLUE *= 0.9.
    Return sum of all transformed channel values.

    Args:
        n: Number of pixels to process.

    Returns:
        Sum of all transformed RGB values.
    """
    total = 0
    for i in range(n):
        r = (i * 41 + 7) % 256
        g = (i * 59 + 13) % 256
        b = (i * 71 + 3) % 256

        r_new = int(r * 0.8)
        g_new = int(g * 1.1)
        if g_new > 255:
            g_new = 255
        b_new = int(b * 0.9)

        total += r_new + g_new + b_new

    return total
