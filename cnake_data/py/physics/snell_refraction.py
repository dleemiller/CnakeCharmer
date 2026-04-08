"""Multi-layer Snell's law refraction through a planar optical stack.

Traces a bundle of rays through alternating media layers using Snell's law
(n1 sin θ1 = n2 sin θ2). Computes total path length and exit angle statistics.

Keywords: Snell's law, refraction, optical stack, ray tracing, physical optics
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000, 10))
def snell_refraction(n_rays: int, n_layers: int) -> tuple:
    """Trace n_rays through n_layers planar interfaces using Snell's law.

    Ray bundle enters from air (n=1.0) at angles uniformly spanning [0, 45°].
    Layers alternate between glass (n=1.5) and air (n=1.0).

    Args:
        n_rays: Number of rays.
        n_layers: Number of refractive interfaces.

    Returns:
        Tuple of (sum_exit_sin, transmitted_count, total_path_length).
    """
    layer_n = [1.0 if i % 2 == 0 else 1.5 for i in range(n_layers + 1)]
    layer_d = 5.0  # thickness of each layer (mm)

    sum_exit_sin = 0.0
    transmitted = 0
    total_path = 0.0

    for i in range(n_rays):
        theta = math.pi / 4.0 * i / (n_rays - 1) if n_rays > 1 else 0.0
        sin_theta = math.sin(theta)
        path_len = 0.0
        tir = False  # total internal reflection

        for layer in range(n_layers):
            n1 = layer_n[layer]
            n2 = layer_n[layer + 1]
            sin_theta2 = n1 * sin_theta / n2
            if abs(sin_theta2) > 1.0:
                # Total internal reflection
                tir = True
                break
            # Path length through this layer
            cos_t = math.sqrt(1.0 - sin_theta2 * sin_theta2)
            path_len += layer_d / cos_t if cos_t > 0.0 else 0.0
            sin_theta = sin_theta2

        if not tir:
            sum_exit_sin += sin_theta
            transmitted += 1
            total_path += path_len

    return (sum_exit_sin, transmitted, total_path)
