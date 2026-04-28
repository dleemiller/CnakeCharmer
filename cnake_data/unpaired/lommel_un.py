import numpy as np
from scipy.special import jv


def un_cy03(u, v, n, s=20):
    """Lommel function Un approximation via finite summation."""
    l = len(u)
    uval = np.zeros((l,), dtype=np.float64)
    for i in range(l):
        for j in range(s + 1):
            ui = u[i]
            vi = v[i]
            jn = jv(n + 2 * j, vi)
            uval[i] += ((-1) ** j) * ((ui / vi) ** (n + 2 * j)) * jn
    return np.asarray(uval)
