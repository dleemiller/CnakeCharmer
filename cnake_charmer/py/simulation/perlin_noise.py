"""Perlin noise with fractal Brownian motion.

Keywords: perlin noise, fbm, procedural generation, noise, simulation
"""

import math

from cnake_charmer.benchmarks import python_benchmark

_p_base = [
    151,
    160,
    137,
    91,
    90,
    15,
    131,
    13,
    201,
    95,
    96,
    53,
    194,
    233,
    7,
    225,
    140,
    36,
    103,
    30,
    69,
    142,
    8,
    99,
    37,
    240,
    21,
    10,
    23,
    190,
    6,
    148,
    247,
    120,
    234,
    75,
    0,
    26,
    197,
    62,
    94,
    252,
    219,
    203,
    117,
    35,
    11,
    32,
    57,
    177,
    33,
    88,
    237,
    149,
    56,
    87,
    174,
    20,
    125,
    136,
    171,
    168,
    68,
    175,
    74,
    165,
    71,
    134,
    139,
    48,
    27,
    166,
    77,
    146,
    158,
    231,
    83,
    111,
    229,
    122,
    60,
    211,
    133,
    230,
    220,
    105,
    92,
    41,
    55,
    46,
    245,
    40,
    244,
    102,
    143,
    54,
    65,
    25,
    63,
    161,
    1,
    216,
    80,
    73,
    209,
    76,
    132,
    187,
    208,
    89,
    18,
    169,
    200,
    196,
    135,
    130,
    116,
    188,
    159,
    86,
    164,
    100,
    109,
    198,
    173,
    186,
    3,
    64,
    52,
    217,
    226,
    250,
    124,
    123,
    5,
    202,
    38,
    147,
    118,
    126,
    255,
    82,
    85,
    212,
    207,
    206,
    59,
    227,
    47,
    16,
    58,
    17,
    182,
    189,
    28,
    42,
    223,
    183,
    170,
    213,
    119,
    248,
    152,
    2,
    44,
    154,
    163,
    70,
    221,
    153,
    101,
    155,
    167,
    43,
    172,
    9,
    129,
    22,
    39,
    253,
    19,
    98,
    108,
    110,
    79,
    113,
    224,
    232,
    178,
    185,
    112,
    104,
    218,
    246,
    97,
    228,
    251,
    34,
    242,
    193,
    238,
    210,
    144,
    12,
    191,
    179,
    162,
    241,
    81,
    51,
    145,
    235,
    249,
    14,
    239,
    107,
    49,
    192,
    214,
    31,
    181,
    199,
    106,
    157,
    184,
    84,
    204,
    176,
    115,
    121,
    50,
    45,
    127,
    4,
    150,
    254,
    138,
    236,
    205,
    93,
    222,
    114,
    67,
    29,
    24,
    72,
    243,
    141,
    128,
    195,
    78,
    66,
    215,
    61,
    156,
    180,
]
_p = _p_base + _p_base


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(t, a, b):
    return a + t * (b - a)


def _grad(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    r = u if (h & 1) == 0 else -u
    r += v if (h & 2) == 0 else -v
    return r


def _noise(x, y, z):
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    zi = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)

    a = _p[xi] + yi
    aa = _p[a] + zi
    ab = _p[a + 1] + zi
    b = _p[xi + 1] + yi
    ba = _p[b] + zi
    bb = _p[b + 1] + zi

    return _lerp(
        w,
        _lerp(
            v,
            _lerp(u, _grad(_p[aa], x, y, z), _grad(_p[ba], x - 1, y, z)),
            _lerp(u, _grad(_p[ab], x, y - 1, z), _grad(_p[bb], x - 1, y - 1, z)),
        ),
        _lerp(
            v,
            _lerp(u, _grad(_p[aa + 1], x, y, z - 1), _grad(_p[ba + 1], x - 1, y, z - 1)),
            _lerp(u, _grad(_p[ab + 1], x, y - 1, z - 1), _grad(_p[bb + 1], x - 1, y - 1, z - 1)),
        ),
    )


def _fbm(x, y, z, octaves=6):
    amplitude = 1.0
    frequency = 1.0
    accum = 0.0
    for _ in range(octaves):
        accum += amplitude * _noise(x * frequency, y * frequency, z * frequency)
        amplitude *= 0.5
        frequency *= 2.0
    return accum


@python_benchmark(args=(80,))
def perlin_noise(n):
    """Evaluate FBM Perlin noise on an n×n grid.

    Args:
        n: Grid dimension.

    Returns:
        Tuple of (total_sum, min_val, max_val, sample_at_half).
    """
    total = 0.0
    min_val = 1e300
    max_val = -1e300
    sample_val = 0.0
    half = n // 2

    for i in range(n):
        for j in range(n):
            val = _fbm(i * 0.1, j * 0.1, 0.5)
            total += val
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            if i == half and j == half:
                sample_val = val

    return (total, min_val, max_val, sample_val)
