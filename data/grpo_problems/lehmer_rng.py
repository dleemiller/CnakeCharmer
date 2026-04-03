def lehmer_rng(seed, n):
    """Generate n pseudo-random numbers in [0, 1] using Lehmer's modular generator.

    Uses the classic Lehmer (Park-Miller) linear congruential generator with
    multiplier a=16807 and modulus m=2^31-1.

    Args:
        seed: initial seed value (positive float or int)
        n: number of random values to generate

    Returns:
        A list of n floats in [0, 1).
    """
    a = 16807.0
    xm = 2147483647.0
    x0 = 2147483711.0

    result = []
    s = float(seed)
    for _i in range(n):
        s = (a * s) % xm
        result.append(s / x0)

    return result
