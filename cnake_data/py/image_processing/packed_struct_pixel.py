"""Process RGBA pixels with packed struct layout.

Demonstrates packed struct with no padding between fields.
Applies gamma correction and alpha blending to n pixels.

Keywords: image processing, pixel, packed struct, RGBA, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def packed_struct_pixel(n: int) -> int:
    """Process n RGBA pixels, return checksum.

    Applies simple gamma adjustment and alpha premultiplication.

    Args:
        n: Number of pixels.

    Returns:
        Checksum of processed pixel values.
    """
    mask = 0xFFFFFFFF
    checksum = 0

    for i in range(n):
        h = (i * 2654435761) & mask
        r = h & 0xFF
        g = (h >> 8) & 0xFF
        b = (h >> 16) & 0xFF
        a = (h >> 24) & 0xFF

        # Simple gamma: square root approximation
        # Using integer approximation: val * val // 255
        r = (r * r) // 255
        g = (g * g) // 255
        b = (b * b) // 255

        # Alpha premultiply
        r = (r * a) // 255
        g = (g * a) // 255
        b = (b * a) // 255

        checksum += r + g + b + a

    return checksum & mask
