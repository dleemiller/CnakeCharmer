"""Compute rectangle intersection areas using nested structs.

Demonstrates nested struct pattern: Point inside Rect.

Keywords: geometry, rectangle, intersection, nested struct, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def nested_struct_rect(n: int) -> float:
    """Compute total intersection area of n rectangle pairs.

    Args:
        n: Number of rectangle pairs to test.

    Returns:
        Sum of intersection areas.
    """
    mask = 0xFFFFFFFF
    total_area = 0.0

    for i in range(n):
        # Generate two rectangles from hash
        h1 = (i * 2654435761) & mask
        h2 = (i * 2246822519) & mask
        h3 = ((i + 1) * 2654435761) & mask
        h4 = ((i + 1) * 2246822519) & mask

        # Rect A
        ax0 = (h1 & 0xFFFF) / 65535.0 * 100.0
        ay0 = ((h1 >> 16) & 0xFFFF) / 65535.0 * 100.0
        ax1 = ax0 + (h2 & 0xFFFF) / 65535.0 * 20.0 + 1.0
        ay1 = ay0 + ((h2 >> 16) & 0xFFFF) / 65535.0 * 20.0 + 1.0

        # Rect B
        bx0 = (h3 & 0xFFFF) / 65535.0 * 100.0
        by0 = ((h3 >> 16) & 0xFFFF) / 65535.0 * 100.0
        bx1 = bx0 + (h4 & 0xFFFF) / 65535.0 * 20.0 + 1.0
        by1 = by0 + ((h4 >> 16) & 0xFFFF) / 65535.0 * 20.0 + 1.0

        # Intersection
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        if ix0 < ix1 and iy0 < iy1:
            total_area += (ix1 - ix0) * (iy1 - iy0)

    return total_area
