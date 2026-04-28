import math

import numpy as np


def distance(pt1, pt2):
    total = 0.0
    for i in range(len(pt1)):
        total += (pt1[i] - pt2[i]) ** 2
    return math.sqrt(total)


def compute_select_becke(points, atoms, radii, select, order):
    m = atoms.shape[0]
    n = points.shape[0]

    alpha = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            u_ab = (radii[i] - radii[j]) / (radii[i] + radii[j])
            u_ab /= u_ab**2 - 1
            u_ab = max(min(u_ab, 0.45), -0.45)
            alpha[i, j] = u_ab
            alpha[j, i] = -u_ab

    at_dists = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            at_dists[i, j] = distance(atoms[i], atoms[j])
            at_dists[j, i] = at_dists[i, j]

    becke_wts = np.ones(n, dtype=float)
    for i in range(n):
        nom = 0.0
        denom = 0.0
        for j in range(m):
            p = 1.0
            for k in range(m):
                if j == k:
                    continue
                miu = (distance(points[i], atoms[j]) - distance(points[i], atoms[k])) / at_dists[
                    j, k
                ]
                nu = miu + alpha[j, k] * (1 - miu**2)
                for _ in range(order):
                    nu = 0.5 * nu * (3 - nu**2)
                s = 0.5 * (1 - nu)
                p *= s
            denom += p
            if j == select:
                nom = p
        becke_wts[i] *= nom / denom
    return becke_wts


def compute_becke_weights(points, atoms, radii, selects, pt_ind, order=3):
    m = selects.shape[0]
    n = points.shape[0]
    tot_wts = np.zeros(n, dtype=float)
    for i in range(m):
        ind1 = pt_ind[i]
        ind2 = pt_ind[i + 1]
        tot_wts[ind1:ind2] = compute_select_becke(
            points[ind1:ind2], atoms, radii, int(selects[i]), order
        )
    return tot_wts
