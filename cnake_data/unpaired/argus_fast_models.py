import numpy as np


def c_min(arr):
    arr_min = 1e12
    for v in arr:
        if v < arr_min:
            arr_min = v
    return arr_min


def fast_gauss(arr, x, y, tau):
    n_arr = arr.shape[0]
    gauss = np.empty(n_arr, dtype=float)
    for idx in range(n_arr):
        dist2 = (arr[idx, 0] - x) ** 2 + (arr[idx, 1] - y) ** 2
        gauss[idx] = np.exp(-dist2 / (2.0 * (tau**2)))
    return gauss


def fast_jansonius(rho, phi0, beta_s, beta_i):
    deg2rad = np.pi / 180.0
    if phi0 > 0:
        b = np.exp(beta_s + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
        c = 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
    else:
        b = -np.exp(beta_i + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
        c = 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)

    xprime = np.empty_like(rho, dtype=float)
    yprime = np.empty_like(rho, dtype=float)
    rho_min = c_min(rho)

    for idx in range(len(rho)):
        tmp_rho = rho[idx]
        tmp_phi = phi0 + b * ((tmp_rho - rho_min) ** c)
        xprime[idx] = tmp_rho * np.cos(deg2rad * tmp_phi)
        yprime[idx] = tmp_rho * np.sin(deg2rad * tmp_phi)

    return xprime, yprime


def argmin_segment(bundles, x, y):
    min_dist2 = 1e12
    min_seg = 0
    n_seg = bundles.shape[0]
    for seg in range(n_seg):
        dist2 = (bundles[seg, 0] - x) ** 2 + (bundles[seg, 1] - y) ** 2
        if dist2 < min_dist2:
            min_dist2 = dist2
            min_seg = seg
    return min_seg


def fast_finds_closest_axons(bundles, xret, yret):
    closest_seg = np.empty(len(xret), dtype=int)
    for i in range(len(xret)):
        closest_seg[i] = argmin_segment(bundles, xret[i], yret[i])
    return closest_seg
