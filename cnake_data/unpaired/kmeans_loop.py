import numpy as np


def kmeansloop_cython(x, init, maxiter=100, tol=1e-4, weight=None):
    l = x.shape[0]
    n = x.shape[1]
    k = init.shape[0]

    oldcenter = np.zeros((k, n), dtype=float)
    newcenter = np.zeros((k, n), dtype=float)
    count = 0

    if weight is None:
        weight = np.ones(l)

    for i in range(k):
        for j in range(n):
            oldcenter[i, j] = init[i, j]

    while count < maxiter:
        count += 1
        clulist = np.zeros(l, dtype=int)
        dislist = np.zeros(l, dtype=float)
        dist = np.zeros(k, dtype=float)

        for i in range(l):
            for j in range(k):
                temp = oldcenter[j] - x[i]
                dist[j] = np.linalg.norm(temp)
            best = int(np.argmin(dist))
            clulist[i] = best
            dislist[i] = dist[best]

        cluscenter = np.unique(clulist)
        newcenter[:, :] = oldcenter[:, :]

        for i in range(k):
            if i in cluscenter:
                for j in range(n):
                    num = 0.0
                    a = 0.0
                    for h in range(l):
                        if clulist[h] == i:
                            num += weight[h] * x[h, j]
                            a += weight[h]
                    newcenter[i, j] = num / a

        if np.allclose(oldcenter, newcenter, rtol=0, atol=tol):
            break

        oldcenter[:, :] = newcenter[:, :]

    cost = np.inner(weight, np.square(dislist))
    return newcenter, clulist, count, cost
