# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Convert WGS84 latitude/longitude to British National Grid (OSGB36) eastings/northings.

Performs a 7-parameter Helmert transformation from the GRS80 ellipsoid to
Airy 1830, then projects via Transverse Mercator to produce grid coordinates.

Keywords: geometry, coordinate transform, geodesy, Helmert, Transverse Mercator, cython, benchmark
"""

from libc.math cimport sqrt, sin, cos, tan, atan2, M_PI

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def wgs84_to_osgb36(int n):
    """Convert n evenly-spaced UK lat/lon pairs to OSGB36 eastings/northings."""
    cdef double sum_e = 0.0
    cdef double sum_n = 0.0
    cdef int count = 0
    cdef int i
    cdef double lat, lon, lat_1, lon_1
    cdef double a_1, b_1, e2_1, nu_1, H
    cdef double x_1, y_1, z_1, s
    cdef double tx, ty, tz, rxs, rys, rzs, rx, ry, rz
    cdef double x_2, y_2, z_2
    cdef double a, b, e2, p, lat_r, lon_r, nu
    cdef double F0, lat0, lon0, N0, E0, nn
    cdef double rho, eta2
    cdef double M1, M2, M3, M4, M
    cdef double I_val, II, III, IIIA, IV, V, VI
    cdef double dl, N_val, E_val
    cdef double sin_lat, cos_lat, tan_lat
    cdef int j

    for i in range(n):
        lat = 50.0 + (i % 100) * 0.08
        lon = -6.0 + (i / 100) * 0.06

        lat_1 = lat * M_PI / 180.0
        lon_1 = lon * M_PI / 180.0

        a_1 = 6378137.000
        b_1 = 6356752.3141
        e2_1 = 1.0 - (b_1 * b_1) / (a_1 * a_1)
        nu_1 = a_1 / sqrt(1.0 - e2_1 * sin(lat_1) * sin(lat_1))

        H = 0.0
        x_1 = (nu_1 + H) * cos(lat_1) * cos(lon_1)
        y_1 = (nu_1 + H) * cos(lat_1) * sin(lon_1)
        z_1 = ((1.0 - e2_1) * nu_1 + H) * sin(lat_1)

        s = 20.4894e-6
        tx = -446.448
        ty = 125.157
        tz = -542.060
        rxs = -0.1502
        rys = -0.2470
        rzs = -0.8421
        rx = rxs * M_PI / (180.0 * 3600.0)
        ry = rys * M_PI / (180.0 * 3600.0)
        rz = rzs * M_PI / (180.0 * 3600.0)

        x_2 = tx + (1.0 + s) * x_1 + (-rz) * y_1 + ry * z_1
        y_2 = ty + rz * x_1 + (1.0 + s) * y_1 + (-rx) * z_1
        z_2 = tz + (-ry) * x_1 + rx * y_1 + (1.0 + s) * z_1

        a = 6377563.396
        b = 6356256.909
        e2 = 1.0 - (b * b) / (a * a)
        p = sqrt(x_2 * x_2 + y_2 * y_2)

        lat_r = atan2(z_2, p * (1.0 - e2))
        for j in range(10):
            nu = a / sqrt(1.0 - e2 * sin(lat_r) * sin(lat_r))
            lat_r = atan2(z_2 + e2 * nu * sin(lat_r), p)

        lon_r = atan2(y_2, x_2)

        F0 = 0.9996012717
        lat0 = 49.0 * M_PI / 180.0
        lon0 = -2.0 * M_PI / 180.0
        N0 = -100000.0
        E0 = 400000.0
        nn = (a - b) / (a + b)

        nu = a * F0 / sqrt(1.0 - e2 * sin(lat_r) * sin(lat_r))
        rho = a * F0 * (1.0 - e2) / ((1.0 - e2 * sin(lat_r) * sin(lat_r)) * sqrt(1.0 - e2 * sin(lat_r) * sin(lat_r)))
        eta2 = nu / rho - 1.0

        sin_lat = sin(lat_r)
        cos_lat = cos(lat_r)
        tan_lat = tan(lat_r)

        M1 = (1.0 + nn + 1.25 * nn * nn + 1.25 * nn * nn * nn) * (lat_r - lat0)
        M2 = (3.0 * nn + 3.0 * nn * nn + 2.625 * nn * nn * nn) * sin(lat_r - lat0) * cos(lat_r + lat0)
        M3 = (1.875 * nn * nn + 1.875 * nn * nn * nn) * sin(2.0 * (lat_r - lat0)) * cos(2.0 * (lat_r + lat0))
        M4 = (35.0 / 24.0) * nn * nn * nn * sin(3.0 * (lat_r - lat0)) * cos(3.0 * (lat_r + lat0))
        M = b * F0 * (M1 - M2 + M3 - M4)

        I_val = M + N0
        II = nu * sin_lat * cos_lat / 2.0
        III = nu * sin_lat * cos_lat * cos_lat * cos_lat * (5.0 - tan_lat * tan_lat + 9.0 * eta2) / 24.0
        IIIA = nu * sin_lat * cos_lat * cos_lat * cos_lat * cos_lat * cos_lat * (61.0 - 58.0 * tan_lat * tan_lat + tan_lat * tan_lat * tan_lat * tan_lat) / 720.0
        IV = nu * cos_lat
        V = nu * cos_lat * cos_lat * cos_lat * (nu / rho - tan_lat * tan_lat) / 6.0
        VI = nu * cos_lat * cos_lat * cos_lat * cos_lat * cos_lat * (5.0 - 18.0 * tan_lat * tan_lat + tan_lat * tan_lat * tan_lat * tan_lat + 14.0 * eta2 - 58.0 * eta2 * tan_lat * tan_lat) / 120.0

        dl = lon_r - lon0
        N_val = I_val + II * dl * dl + III * dl * dl * dl * dl + IIIA * dl * dl * dl * dl * dl * dl
        E_val = E0 + IV * dl + V * dl * dl * dl + VI * dl * dl * dl * dl * dl

        sum_e = sum_e + E_val
        sum_n = sum_n + N_val
        count = count + 1

    return (round(sum_e, 2), round(sum_n, 2), count)
