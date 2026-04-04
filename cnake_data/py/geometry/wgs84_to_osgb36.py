"""Convert WGS84 latitude/longitude to British National Grid (OSGB36) eastings/northings.

Performs a 7-parameter Helmert transformation from the GRS80 ellipsoid to
Airy 1830, then projects via Transverse Mercator to produce grid coordinates.

Keywords: geometry, coordinate transform, geodesy, Helmert, Transverse Mercator, benchmark
"""

from math import atan2, cos, pi, sin, sqrt, tan

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def wgs84_to_osgb36(n: int) -> tuple:
    """Convert n evenly-spaced UK lat/lon pairs to OSGB36 eastings/northings.

    Generates n test coordinates spanning a grid across Great Britain and
    converts each pair, returning the sum of all eastings and northings
    for validation.

    Args:
        n: Number of coordinate pairs to convert.

    Returns:
        Tuple of (sum_eastings, sum_northings, count).
    """
    sum_e = 0.0
    sum_n = 0.0
    count = 0

    for i in range(n):
        # Generate lat/lon spanning Great Britain
        lat = 50.0 + (i % 100) * 0.08  # 50-58 degrees
        lon = -6.0 + (i // 100) * 0.06  # -6 to 0 degrees

        # Convert to radians (GRS80 ellipsoid)
        lat_1 = lat * pi / 180.0
        lon_1 = lon * pi / 180.0

        # GRS80 semi-major/minor axes
        a_1 = 6378137.000
        b_1 = 6356752.3141
        e2_1 = 1 - (b_1 * b_1) / (a_1 * a_1)
        nu_1 = a_1 / sqrt(1 - e2_1 * sin(lat_1) ** 2)

        # Cartesian from spherical polar
        H = 0.0
        x_1 = (nu_1 + H) * cos(lat_1) * cos(lon_1)
        y_1 = (nu_1 + H) * cos(lat_1) * sin(lon_1)
        z_1 = ((1 - e2_1) * nu_1 + H) * sin(lat_1)

        # Helmert transform: GRS80 -> Airy 1830
        s = 20.4894e-6
        tx, ty, tz = -446.448, 125.157, -542.060
        rxs, rys, rzs = -0.1502, -0.2470, -0.8421
        rx = rxs * pi / (180.0 * 3600.0)
        ry = rys * pi / (180.0 * 3600.0)
        rz = rzs * pi / (180.0 * 3600.0)

        x_2 = tx + (1 + s) * x_1 + (-rz) * y_1 + ry * z_1
        y_2 = ty + rz * x_1 + (1 + s) * y_1 + (-rx) * z_1
        z_2 = tz + (-ry) * x_1 + rx * y_1 + (1 + s) * z_1

        # Back to spherical on Airy 1830
        a = 6377563.396
        b = 6356256.909
        e2 = 1 - (b * b) / (a * a)
        p = sqrt(x_2**2 + y_2**2)

        # Iterative latitude
        lat_r = atan2(z_2, p * (1 - e2))
        for _ in range(10):
            nu = a / sqrt(1 - e2 * sin(lat_r) ** 2)
            lat_r = atan2(z_2 + e2 * nu * sin(lat_r), p)

        lon_r = atan2(y_2, x_2)

        # Transverse Mercator projection
        F0 = 0.9996012717
        lat0 = 49.0 * pi / 180.0
        lon0 = -2.0 * pi / 180.0
        N0, E0 = -100000.0, 400000.0
        nn = (a - b) / (a + b)

        nu = a * F0 / sqrt(1 - e2 * sin(lat_r) ** 2)
        rho = a * F0 * (1 - e2) * (1 - e2 * sin(lat_r) ** 2) ** (-1.5)
        eta2 = nu / rho - 1.0

        M1 = (1 + nn + 1.25 * nn**2 + 1.25 * nn**3) * (lat_r - lat0)
        M2 = (3 * nn + 3 * nn**2 + 2.625 * nn**3) * sin(lat_r - lat0) * cos(lat_r + lat0)
        M3 = (1.875 * nn**2 + 1.875 * nn**3) * sin(2 * (lat_r - lat0)) * cos(2 * (lat_r + lat0))
        M4 = (35.0 / 24.0) * nn**3 * sin(3 * (lat_r - lat0)) * cos(3 * (lat_r + lat0))
        M = b * F0 * (M1 - M2 + M3 - M4)

        term_I = M + N0
        II = nu * sin(lat_r) * cos(lat_r) / 2.0
        III = nu * sin(lat_r) * cos(lat_r) ** 3 * (5 - tan(lat_r) ** 2 + 9 * eta2) / 24.0
        IIIA = (
            nu
            * sin(lat_r)
            * cos(lat_r) ** 5
            * (61 - 58 * tan(lat_r) ** 2 + tan(lat_r) ** 4)
            / 720.0
        )
        IV = nu * cos(lat_r)
        V = nu * cos(lat_r) ** 3 * (nu / rho - tan(lat_r) ** 2) / 6.0
        VI = (
            nu
            * cos(lat_r) ** 5
            * (5 - 18 * tan(lat_r) ** 2 + tan(lat_r) ** 4 + 14 * eta2 - 58 * eta2 * tan(lat_r) ** 2)
            / 120.0
        )

        dl = lon_r - lon0
        N_val = term_I + II * dl**2 + III * dl**4 + IIIA * dl**6
        E_val = E0 + IV * dl + V * dl**3 + VI * dl**5

        sum_e += E_val
        sum_n += N_val
        count += 1

    return (round(sum_e, 2), round(sum_n, 2), count)
