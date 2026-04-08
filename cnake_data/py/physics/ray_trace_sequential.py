"""Sequential paraxial ray tracing through a multi-element lens system.

Uses 2×2 ABCD transfer matrices (ray transfer matrix formalism) to trace
a fan of paraxial rays through alternating refraction and propagation surfaces.

Keywords: ray tracing, paraxial optics, ABCD matrix, lens system, geometric optics
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000, 8))
def ray_trace_sequential(n_rays: int, n_surfaces: int) -> tuple:
    """Trace n_rays through an n_surfaces lens system using ABCD matrices.

    Args:
        n_rays: Number of rays (fan of angles from -0.1 to 0.1 rad).
        n_surfaces: Number of optical surfaces (alternating refraction/propagation).

    Returns:
        Tuple of (sum_y, sum_u, n_focused) where y/u are final ray height/angle
        sums and n_focused counts rays that pass the stop aperture |y| < 5.0.
    """
    # Lens parameters: focal length 50mm, diameter 25mm
    # Alternating surfaces: propagation d=10mm, refraction f=50mm
    d = 10.0  # propagation distance (mm)
    f = 50.0  # focal length of each refractive surface

    sum_y = 0.0
    sum_u = 0.0
    n_focused = 0

    for i in range(n_rays):
        # Fan of input rays: vary angle, fixed height y=1.0
        u0 = -0.1 + 0.2 * i / (n_rays - 1) if n_rays > 1 else 0.0
        y = 1.0
        u = u0

        for s in range(n_surfaces):
            if s % 2 == 0:
                # Propagation: y' = y + d*u, u' = u
                y = y + d * u
            else:
                # Thin lens refraction: y' = y, u' = u - y/f
                u = u - y / f

        sum_y += y
        sum_u += u
        if abs(y) < 5.0:
            n_focused += 1

    return (sum_y, sum_u, n_focused)
