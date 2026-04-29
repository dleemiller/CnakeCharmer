"""2D polynomial multiply/integrate/derivative utilities."""

from __future__ import annotations


def polymul2d(a, b, c):
    for n in range(len(c)):
        for m in range(len(c[0])):
            acc = 0.0
            l0 = max(n - len(a) + 1, 0)
            l1 = min(n + 1, len(b))
            l20 = max(m - len(a[0]) + 1, 0)
            l21 = min(m + 1, len(b[0]))
            for l in range(l0, l1):
                for l2 in range(l20, l21):
                    k = n - l
                    k2 = m - l2
                    acc += a[k][k2] * b[l][l2]
            c[n][m] = acc
    return c


def polyint2d(a, b, xmin, xmax, ymin, ymax):
    for l in range(len(a[0])):
        b[l] = 0.0
        for k in range(len(a)):
            b[l] += ((xmax ** (k + 1) - xmin ** (k + 1)) / float(k + 1)) * a[k][l]

    out = 0.0
    for l in range(len(a[0])):
        out += ((ymax ** (l + 1) - ymin ** (l + 1)) / float(l + 1)) * b[l]
    return out


def polyder2d(a, a_out, axis):
    rows = len(a)
    cols = len(a[0])
    if axis == 0:
        for i in range(rows - 1):
            for j in range(cols):
                a_out[i][j] = (i + 1) * a[i + 1][j]
        for j in range(cols):
            a_out[rows - 1][j] = 0.0
    elif axis == 1:
        for i in range(rows):
            for j in range(cols - 1):
                a_out[i][j] = (j + 1) * a[i][j + 1]
        for i in range(rows):
            a_out[i][cols - 1] = 0.0
    return a_out
