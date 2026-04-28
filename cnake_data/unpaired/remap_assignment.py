import numpy as np

DTYPE = np.float64
DTYPEINT = np.int64


def assignment(f_old, map1, map2, dim1, dim2):
    f_new = np.zeros((dim1, dim2), dtype=DTYPE)
    for i in range(dim1):
        for j in range(dim2):
            f_new[map1[i, j], map2[i, j]] += f_old[i, j]
    return f_new


def sift(f, cfl):
    dim1, dim2 = f.shape
    f_nonneg = np.zeros((dim1, dim2), dtype=DTYPE)
    f_neg = np.zeros((dim1, dim2), dtype=DTYPE)

    for j in range(dim2):
        if cfl[0, j] >= 0:
            for i in range(dim1):
                f_nonneg[i, j] = f[i, j]
        else:
            for i in range(dim1):
                f_neg[i, j] = f[i, j]
    return f_nonneg, f_neg
