import math


def cal_delta(delta, ln_s_jt, mu, etol=1e-9, iter_limit=1000):
    """Contraction mapping update for BLP-style mean utility.

    Shapes:
        delta: [nmkts][nbrands]
        ln_s_jt: [nmkts][nbrands]
        mu: [nmkts][nsiminds][nbrands]
    """
    nmkts = len(mu)
    nsiminds = len(mu[0]) if nmkts else 0
    nbrands = len(mu[0][0]) if nsiminds else 0

    niter = 0
    while True:
        diff_max = 0.0
        diff_mean = 0.0

        for mkt in range(nmkts):
            mktshr = [0.0] * nbrands

            for ind in range(nsiminds):
                exp_xb = [0.0] * nbrands
                denom = 1.0
                for brand in range(nbrands):
                    ex = math.exp(delta[mkt][brand] + mu[mkt][ind][brand])
                    exp_xb[brand] = ex
                    denom += ex
                for brand in range(nbrands):
                    mktshr[brand] += exp_xb[brand] / (denom * nsiminds)

            for brand in range(nbrands):
                diff = ln_s_jt[mkt][brand] - math.log(mktshr[brand])
                delta[mkt][brand] += diff
                ad = abs(diff)
                if ad > diff_max:
                    diff_max = ad
                diff_mean += diff

        diff_mean /= max(nmkts * nbrands, 1)

        if (diff_max < etol and diff_mean < 1e-3) or niter > iter_limit:
            break
        niter += 1

    return delta, niter


def cal_s(delta, mu):
    """Compute market shares by simulation integration."""
    nmkts = len(mu)
    nsiminds = len(mu[0]) if nmkts else 0
    nbrands = len(mu[0][0]) if nsiminds else 0

    s = [[0.0] * nbrands for _ in range(nmkts)]

    for mkt in range(nmkts):
        for ind in range(nsiminds):
            exp_xb = [0.0] * nbrands
            denom = 1.0
            for brand in range(nbrands):
                ex = math.exp(delta[mkt][brand] + mu[mkt][ind][brand])
                exp_xb[brand] = ex
                denom += ex
            for brand in range(nbrands):
                s[mkt][brand] += exp_xb[brand] / (denom * nsiminds)

    return s
