import numpy as np

DTYPE = np.double


def argmin_abs(vec, elem):
    min_val = np.inf
    minidx = 0
    for i in range(len(vec)):
        d = abs(vec[i] - elem)
        if d < min_val:
            min_val = d
            minidx = i
    return minidx


def vis_weighting(daz_vec, del_vec, daz_area, ant_weight, minvisvals, el):
    ndaz = len(daz_vec)
    ndel = len(del_vec)
    vis = 0.0

    for idaz in range(ndaz):
        mina = argmin_abs(daz_area, daz_vec[idaz])
        for idel in range(ndel):
            if minvisvals[mina] <= el + del_vec[idel]:
                vis += ant_weight[idaz, idel]
    return vis
