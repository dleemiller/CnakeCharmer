import math


def volumetric_parametric_eval(num_h, x_num, x_min, x_max, h_min, h_max):
    """Evaluate a nested 3D expression across a parameterized h-grid."""
    out = [0.0 for _ in range(num_h)]
    if num_h <= 1 or x_num <= 1:
        return out

    for h_i in range(num_h):
        h = h_min + (h_max - h_min) * h_i / (num_h - 1)
        result = 0.0

        for x_i in range(x_num):
            x = x_min + (x_max - x_min) * x_i / (x_num - 1)
            for y_i in range(x_num):
                y = x_min + (x_max - x_min) * y_i / (x_num - 1)
                y2 = y * y
                for z_i in range(x_num):
                    z = x_min + (x_max - x_min) * z_i / (x_num - 1)
                    t = 2.0 * z + x + y2 - h
                    u = y + z + h
                    result += math.exp(-(t * t)) * (math.sin(x + y + 3.0 * z + h) + u * u)

        out[h_i] = result

    return out
