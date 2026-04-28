def mandelbrot_escape(c, n):
    """Mandelbrot escape iteration count, or -1 if bounded through n steps."""
    z = c
    for i in range(n):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4.0:
            return i
    return -1


def julia_escape(z0, c, n):
    """Julia-set escape iteration count, or -1 if bounded through n steps."""
    zx = z0.real
    zy = z0.imag
    cx = c.real
    cy = c.imag

    for i in range(n):
        zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
        if zx * zx + zy * zy >= 4.0:
            return i
    return -1


def generate_mandelbrot(xs, ys, n):
    """Generate an escape-count grid for the Mandelbrot set."""
    m = len(ys)
    k = len(xs)
    grid = [[0 for _ in range(k)] for _ in range(m)]
    for j in range(m):
        for i in range(k):
            z = complex(xs[i], ys[j])
            grid[j][i] = mandelbrot_escape(z, n)
    return grid


def generate_julia(xs, ys, c, n):
    """Generate an escape-count grid for a Julia set with constant c."""
    m = len(ys)
    k = len(xs)
    grid = [[0 for _ in range(k)] for _ in range(m)]
    for j in range(m):
        for i in range(k):
            z0 = complex(xs[i], ys[j])
            grid[j][i] = julia_escape(z0, c, n)
    return grid
