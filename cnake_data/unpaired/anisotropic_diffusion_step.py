from __future__ import annotations


def anisotropic_diffusion_step(
    image: list[list[float]], kappa: float, dt: float
) -> list[list[float]]:
    """One Perona-Malik style diffusion step on a 2D scalar image."""
    h = len(image)
    if h == 0:
        return []
    w = len(image[0])
    out = [row[:] for row in image]

    def c(d: float) -> float:
        # conduction coefficient
        x = d / max(kappa, 1e-12)
        return 1.0 / (1.0 + x * x)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = image[y][x]
            dn = image[y - 1][x] - center
            ds = image[y + 1][x] - center
            dw = image[y][x - 1] - center
            de = image[y][x + 1] - center
            flux = c(dn) * dn + c(ds) * ds + c(dw) * dw + c(de) * de
            out[y][x] = center + dt * flux
    return out
