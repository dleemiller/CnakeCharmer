"""Process config structs that auto-convert to dicts.

Demonstrates struct-to-dict auto-conversion pattern.
Creates n configs with width, height, scale and computes
a weighted area.

Keywords: algorithms, struct, dict, auto-conversion, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def struct_to_dict(n: int) -> float:
    """Process n configs and return total weighted area.

    Each config has width, height, scale. Weighted area
    is width * height * scale.

    Args:
        n: Number of configs to process.

    Returns:
        Sum of weighted areas.
    """
    total = 0.0

    for i in range(n):
        h = ((i * 2654435761) ^ (i * 2246822519)) & 0xFFFFFFFF
        width = (h & 0xFFF) + 1
        height = ((h >> 12) & 0xFFF) + 1
        scale = ((h >> 24) & 0xFF) / 255.0 + 0.1

        config = {"width": width, "height": height, "scale": scale}
        total += config["width"] * config["height"] * config["scale"]

    return total
