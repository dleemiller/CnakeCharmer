import math


def angle_wrap(theta):
    """Wrap an angle in radians to the range [-pi, pi].

    Repeatedly adds or subtracts 2*pi until the angle falls within
    the canonical range.

    Args:
        theta: angle in radians (float).

    Returns:
        The equivalent angle in [-pi, pi].
    """
    pi = math.pi
    two_pi = 2.0 * pi
    while theta < -pi:
        theta += two_pi
    while theta > pi:
        theta -= two_pi
    return theta
