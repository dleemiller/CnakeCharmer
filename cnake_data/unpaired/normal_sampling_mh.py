import math
import random


def rand_uniform():
    return random.random()


def rand_double(min_value, max_value):
    return (max_value - min_value) * rand_uniform() + min_value


def logp(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * z * z


def normal_pdf(x, mu, sigma):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(logp(x, mu, sigma)) / sigma


def rand_normal():
    u1 = 0.0
    while u1 == 0.0:
        u1 = rand_uniform()
    u2 = rand_uniform()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def normal_bm(n_samples):
    out = [0.0] * n_samples
    for i in range(n_samples):
        out[i] = rand_normal()
    return out


def normal_rejection(n_samples):
    out = [0.0] * n_samples
    for i in range(n_samples):
        while True:
            x = rand_double(-3.0, 3.0)
            y = rand_uniform()
            z = normal_pdf(x, 0.0, 1.0)
            if y <= z:
                out[i] = x
                break
    return out


def normal_mh(starts, n_samples):
    n_starts = len(starts)
    samples = [[0.0] * n_samples for _ in range(n_starts)]
    for i in range(n_starts):
        x = starts[i]
        lpx = logp(x, 0.0, 1.0)
        for j in range(n_samples):
            xc = x + 0.2 * rand_normal()
            lpxc = logp(xc, 0.0, 1.0)
            if rand_uniform() < math.exp(lpxc - lpx):
                x = xc
                lpx = lpxc
            samples[i][j] = x
    return samples
