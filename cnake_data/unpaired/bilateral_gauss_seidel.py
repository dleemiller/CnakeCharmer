import math


def optimize_loop_bilateral_horizon(
    img_dis,
    color_weight_matrix,
    gaussian_weight,
    coefficient,
    alpha,
    exclusion,
    size,
):
    """One bilateral Gauss-Seidel sweep using horizontal coefficient terms."""
    error = 0.0
    h, w = size

    for i in range(exclusion, h - exclusion - 1):
        for j in range(exclusion, w - exclusion - 1):
            sub_img = [
                row[j - exclusion : j + exclusion + 1]
                for row in img_dis[i - exclusion : i + exclusion + 1]
            ]
            color_weight = color_weight_matrix[i - exclusion][j - exclusion]

            a = -(
                coefficient[exclusion][exclusion]
                - (coefficient[exclusion][exclusion + 1] + coefficient[exclusion][exclusion - 1])
                / 2.0
            )
            b = sub_img[exclusion][exclusion] - (
                (coefficient[exclusion][exclusion + 1] - coefficient[exclusion][exclusion - 1])
                / 2.0
                / (
                    -2.0 * coefficient[exclusion][exclusion]
                    + coefficient[exclusion][exclusion + 1]
                    + coefficient[exclusion][exclusion - 1]
                )
            )

            weighted_sum = 0.0
            weight_sum = 0.0
            span = 2 * exclusion + 1
            for r in range(span):
                for c in range(span):
                    wgt = gaussian_weight[r][c] * color_weight[r][c]
                    weighted_sum += wgt * sub_img[r][c]
                    weight_sum += wgt

            d_new = (-a * b + weighted_sum) / (-a + weight_sum)
            error += abs(img_dis[i][j] - d_new)
            img_dis[i][j] = d_new

    return img_dis, error


def optimize_loop_bilateral_vertical(
    img_dis,
    color_weight_matrix,
    gaussian_weight,
    coefficient,
    alpha,
    exclusion,
    size,
):
    """One bilateral Gauss-Seidel sweep using vertical coefficient terms."""
    error = 0.0
    h, w = size

    for i in range(exclusion, h - exclusion - 1):
        for j in range(exclusion, w - exclusion - 1):
            sub_img = [
                row[j - exclusion : j + exclusion + 1]
                for row in img_dis[i - exclusion : i + exclusion + 1]
            ]
            color_weight = color_weight_matrix[i - exclusion][j - exclusion]

            a = -(
                coefficient[exclusion][exclusion]
                - (coefficient[exclusion + 1][exclusion] + coefficient[exclusion - 1][exclusion])
                / 2.0
            )
            b = sub_img[exclusion][exclusion] - (
                (coefficient[exclusion + 1][exclusion] - coefficient[exclusion - 1][exclusion])
                / 2.0
                / (
                    -2.0 * coefficient[exclusion][exclusion]
                    + coefficient[exclusion + 1][exclusion]
                    + coefficient[exclusion - 1][exclusion]
                )
            )

            weighted_sum = 0.0
            weight_sum = 0.0
            span = 2 * exclusion + 1
            for r in range(span):
                for c in range(span):
                    wgt = gaussian_weight[r][c] * color_weight[r][c]
                    weighted_sum += wgt * sub_img[r][c]
                    weight_sum += wgt

            d_new = (-a * b + weighted_sum) / (-a + weight_sum)
            error += abs(img_dis[i][j] - d_new)
            img_dis[i][j] = d_new

    return img_dis, error


def make_weight(guide_img, exclusion, size, sigma):
    """Build spatial Gaussian and per-pixel color weight windows."""
    h, w = size
    span = 2 * exclusion + 1

    gaussian_weight = [[0.0 for _ in range(span)] for _ in range(span)]
    for i in range(span):
        for j in range(span):
            dist2 = (i - exclusion) ** 2 + (j - exclusion) ** 2
            gaussian_weight[i][j] = math.exp(-(float(dist2)) / (2.0 * sigma[1] ** 2))

    color_weight_matrix = [
        [[[0.0 for _ in range(span)] for _ in range(span)] for _ in range(w - exclusion)]
        for _ in range(h - exclusion)
    ]

    for i in range(exclusion, h - exclusion - 1):
        for j in range(exclusion, w - exclusion - 1):
            center = guide_img[i][j]
            for r in range(span):
                for c in range(span):
                    g = guide_img[i - exclusion + r][j - exclusion + c]
                    delta = center - g
                    color_weight_matrix[i - exclusion][j - exclusion][r][c] = math.exp(
                        -(delta * delta) / (2.0 * sigma[0] ** 2)
                    )

    return gaussian_weight, color_weight_matrix
