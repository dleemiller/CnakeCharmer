"""
Great circle distance computation for pairs of geographic coordinates.

Keywords: great circle, haversine, distance, geography, numerical, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def great_circle(n: int) -> float:
    """Compute the sum of great circle distances for n point pairs.

    Generates deterministic latitude/longitude pairs and computes
    the great circle distance between each consecutive pair using
    the spherical law of cosines.

    Args:
        n: Number of point pairs to compute distances for.

    Returns:
        Sum of all distances in miles.
    """
    radius = 3956.0
    pi_180 = math.pi / 180.0
    total = 0.0

    for i in range(n):
        lat1 = ((i * 7 + 3) % 180 - 90) * pi_180
        lon1 = ((i * 13 + 7) % 360 - 180) * pi_180
        lat2 = ((i * 11 + 5) % 180 - 90) * pi_180
        lon2 = ((i * 17 + 11) % 360 - 180) * pi_180

        a = (math.pi / 2.0) - lat1
        b = (math.pi / 2.0) - lat2
        theta = lon2 - lon1

        c = math.acos(math.cos(a) * math.cos(b) + math.sin(a) * math.sin(b) * math.cos(theta))
        total += radius * c

    return total
