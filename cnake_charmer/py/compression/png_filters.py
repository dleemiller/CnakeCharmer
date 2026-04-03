"""PNG filter undo operations: sub, up, average, paeth.

Operates on scanlines as byte arrays. Applies all four PNG reconstruction
filters to deterministic scanline data and returns checksums of the results.

Keywords: png, filter, sub, up, average, paeth, compression, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _paeth_predictor(a, b, c):
    """Paeth predictor: pick nearest of a, b, c to p = a + b - c."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


def _generate_scanlines(width, height, filter_unit):
    """Generate deterministic scanline data as list of lists of ints (0-255)."""
    row_bytes = width * filter_unit
    scanlines = []
    for y in range(height):
        row = []
        for x in range(row_bytes):
            row.append(((y * 131 + x * 37 + 7) * 73) % 256)
        scanlines.append(row)
    return scanlines


def _filter_sub(scanlines, filter_unit):
    """Undo sub filter: recon[x] = filt[x] + recon[x - filter_unit]."""
    height = len(scanlines)
    row_bytes = len(scanlines[0])
    result = []
    for y in range(height):
        row = [0] * row_bytes
        for x in range(row_bytes):
            filt = scanlines[y][x]
            left = row[x - filter_unit] if x >= filter_unit else 0
            row[x] = (filt + left) & 0xFF
        result.append(row)
    return result


def _filter_up(scanlines):
    """Undo up filter: recon[x] = filt[x] + prior[x]."""
    height = len(scanlines)
    row_bytes = len(scanlines[0])
    result = []
    for y in range(height):
        row = [0] * row_bytes
        for x in range(row_bytes):
            filt = scanlines[y][x]
            above = result[y - 1][x] if y > 0 else 0
            row[x] = (filt + above) & 0xFF
        result.append(row)
    return result


def _filter_average(scanlines, filter_unit):
    """Undo average filter: recon[x] = filt[x] + floor((left + above) / 2)."""
    height = len(scanlines)
    row_bytes = len(scanlines[0])
    result = []
    for y in range(height):
        row = [0] * row_bytes
        for x in range(row_bytes):
            filt = scanlines[y][x]
            left = row[x - filter_unit] if x >= filter_unit else 0
            above = result[y - 1][x] if y > 0 else 0
            row[x] = (filt + ((left + above) >> 1)) & 0xFF
        result.append(row)
    return result


def _filter_paeth(scanlines, filter_unit):
    """Undo paeth filter: recon[x] = filt[x] + paeth(left, above, upper_left)."""
    height = len(scanlines)
    row_bytes = len(scanlines[0])
    result = []
    for y in range(height):
        row = [0] * row_bytes
        for x in range(row_bytes):
            filt = scanlines[y][x]
            left = row[x - filter_unit] if x >= filter_unit else 0
            above = result[y - 1][x] if y > 0 else 0
            upper_left = result[y - 1][x - filter_unit] if x >= filter_unit and y > 0 else 0
            row[x] = (filt + _paeth_predictor(left, above, upper_left)) & 0xFF
        result.append(row)
    return result


def _checksum(rows):
    """Sum all byte values across all rows."""
    total = 0
    for row in rows:
        for v in row:
            total += v
    return total


@python_benchmark(args=(200, 200, 3))
def png_filters(width: int, height: int, filter_unit: int) -> tuple:
    """Apply all 4 PNG reconstruction filters and return checksums.

    Generates deterministic scanline data, then applies sub, up, average,
    and paeth filter undo operations.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        filter_unit: Bytes per pixel (filter unit size).

    Returns:
        Tuple of (checksum_sub, checksum_up, checksum_avg, checksum_paeth).
    """
    scanlines = _generate_scanlines(width, height, filter_unit)

    result_sub = _filter_sub(scanlines, filter_unit)
    result_up = _filter_up(scanlines)
    result_avg = _filter_average(scanlines, filter_unit)
    result_paeth = _filter_paeth(scanlines, filter_unit)

    return (
        _checksum(result_sub),
        _checksum(result_up),
        _checksum(result_avg),
        _checksum(result_paeth),
    )
