"""Run a fixed-point logistic map and summarize occupancy bands.

Keywords: numerical, logistic map, fixed-point, chaos, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _logistic_step(x: int, r: int, scale: int) -> int:
    return (r * x * (scale - x)) // (scale * scale)


@python_benchmark(args=(300000,))
def logistic_lcg_bands(n: int) -> tuple:
    """Iterate fixed-point logistic dynamics and return band counts."""
    scale = 1 << 20
    r = (39 * scale) // 10
    x = 123456
    bins = [0] * 8
    checksum = 0

    for i in range(n):
        x = _logistic_step(x, r, scale)
        b = (x * 8) // scale
        if b > 7:
            b = 7
        bins[b] += 1
        checksum = (checksum + (x ^ (i * 1315423911))) & 0xFFFFFFFF

    return (bins[0], bins[3], bins[7], checksum)
