import math


def update_soil_temp_array(
    soil_temp_array,
    c_abd,
    i_soil_water_content,
    c_soil_layer_depth,
    p_soil_layer_depth,
    c_avt,
    i_soil_surface_temperature,
    xlag=0.8,
):
    """Update soil layer temperatures with damping-depth weighting."""
    xlg1 = 1.0 - xlag
    dp = 1.0 + (2.5 * c_abd / (c_abd + math.exp(6.53 - (5.63 * c_abd))))
    wc = 0.001 * i_soil_water_content / ((0.356 - (0.144 * c_abd)) * c_soil_layer_depth[-1])
    dd = math.exp(math.log(0.5 / dp) * ((1.0 - wc) / (1.0 + wc)) * 2.0) * dp

    z1 = 0.0
    for i in range(len(soil_temp_array)):
        zd = 0.5 * (z1 + p_soil_layer_depth[i]) / dd
        rate = zd / (zd + math.exp(-0.8669 - (2.0775 * zd))) * (c_avt - i_soil_surface_temperature)
        soil_temp_array[i] = xlag * soil_temp_array[i] + xlg1 * (rate + i_soil_surface_temperature)
        z1 = p_soil_layer_depth[i]

    return soil_temp_array
