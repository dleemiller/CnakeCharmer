import math


def get_volume(v1, dist, dim):
    """Volume scale for radius ``dist`` in ``dim`` dimensions."""
    if dist <= 0:
        return 0.0
    return v1 * math.exp(dim * math.log(dist))


def ratio_test(i, v1, v_dic, dim, distances, k_max, d_thr, indices):
    """Adaptive neighborhood test from PAk-style density estimation."""
    if i not in v_dic:
        v_dic[i] = [-1.0] * (k_max + 1)
        for t in (1, 2, 3):
            v_dic[i][t - 1] = get_volume(v1, distances[i][t], dim)

    k = 3
    dk = 0.0
    while k < k_max and dk <= d_thr:
        vi = v_dic[i][k - 1]
        if vi < 0:
            vi = get_volume(v1, distances[i][k], dim)
            v_dic[i][k - 1] = vi

        j = indices[i][k + 1]
        if j not in v_dic:
            v_dic[j] = [-1.0] * (k_max + 1)
            for t in (1, 2, 3):
                v_dic[j][t - 1] = get_volume(v1, distances[j][t], dim)

        vj = v_dic[j][k - 1]
        if vj < 0:
            vj = get_volume(v1, distances[j][k], dim)
            v_dic[j][k - 1] = vj

        if vi > 0 and vj > 0:
            dk = -2.0 * k * (math.log(vi) + math.log(vj) - 2.0 * math.log(vi + vj) + math.log(4.0))

        k += 1

    v_dic[i][k - 1] = get_volume(v1, distances[i][k], dim)
    return k, distances[i][k - 1], v_dic


def get_densities(dim, distances, k_max, d_thr, indices):
    """Compute PAk-style local densities and uncertainty estimates."""
    v1 = math.exp(dim / 2.0 * math.log(math.pi) - math.lgamma((dim + 2.0) / 2.0))
    n = len(distances)

    k_hat = []
    dc = []
    densities = []
    err_densities = []
    v_dic = {}
    rho_min = float("inf")

    for i in range(n):
        k, dc_i, v_dic = ratio_test(i, v1, v_dic, dim, distances, k_max, d_thr, indices)
        kh = k - 1
        k_hat.append(kh)
        dc.append(dc_i)

        rho = math.log(kh) - (math.log(v1) + dim * math.log(dc_i))
        err = math.sqrt((4.0 * kh + 2.0) / (kh * (kh - 1.0))) if kh > 1 else float("inf")

        densities.append(rho)
        err_densities.append(err)
        if rho < rho_min:
            rho_min = rho

    densities = [x - rho_min + 1.0 for x in densities]
    return k_hat, dc, densities, err_densities
