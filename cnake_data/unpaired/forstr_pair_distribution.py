import math


def mul(r, p, nps, ne, x, y, z, lat, radius, step):
    """Accumulate pair-distance histogram under periodic shifts."""
    pair = 0
    mul_factor = 1.0 / (step * step * step)
    end1 = 0

    for sp1 in range(ne):
        begin1 = end1
        end1 += nps[sp1]
        end2 = begin1

        for sp2 in range(sp1, ne):
            begin2 = end2
            end2 += nps[sp2]

            for at1 in range(begin1, end1):
                for at2 in range(begin2, end2):
                    for i in range(-x, x):
                        for j in range(-y, y):
                            for k in range(-z, z):
                                lx = (
                                    p[at1][0]
                                    - p[at2][0]
                                    - i * lat[0][0]
                                    - j * lat[1][0]
                                    - k * lat[2][0]
                                )
                                ly = (
                                    p[at1][1]
                                    - p[at2][1]
                                    - i * lat[0][1]
                                    - j * lat[1][1]
                                    - k * lat[2][1]
                                )
                                lz = (
                                    p[at1][2]
                                    - p[at2][2]
                                    - i * lat[0][2]
                                    - j * lat[1][2]
                                    - k * lat[2][2]
                                )
                                dis = math.sqrt(lx * lx + ly * ly + lz * lz)
                                if dis < 0.1:
                                    continue
                                if dis < radius:
                                    n = int(dis / step)
                                    r[pair][n] += mul_factor / ((n + 1) * (n + 1) * nps[sp1])
            pair += 1

    return r


def min_bond_length(p, n=None):
    if n is None:
        n = len(p)
    min_dis = 10.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            dz = p[i][2] - p[j][2]
            dis = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dis < min_dis:
                min_dis = dis
    return min_dis
