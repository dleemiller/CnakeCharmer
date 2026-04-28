def diffusion_limited_evaporation(deficit_on_top_layers, soil_diffusion_constant):
    """Compute diffusion-limited evaporation.

    This is a small numeric kernel with branch-heavy logic and scalar math,
    suitable for Cython conversion and policy optimization.
    """
    deficit_m = deficit_on_top_layers / 1000.0

    if deficit_m <= 0.0:
        return 8.3 * 1000.0

    if deficit_m < 25.0:
        value = 2.0 * soil_diffusion_constant * soil_diffusion_constant / deficit_m
        return value * 1000.0

    return 0.0
