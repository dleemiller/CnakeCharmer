import math


def param_gamma_arr(r, sigma, n_prevs, scaling):
    """Return alpha/beta arrays for gamma parametrization."""
    var = sigma * sigma
    alphas = []
    betas = []
    for n_prev in n_prevs:
        coeff = r * n_prev * math.exp(-n_prev / scaling)
        alpha = 1.0 / var
        beta = math.exp(math.log(coeff) + var / 2.0) / alpha
        alphas.append(alpha)
        betas.append(beta)
    return [alphas, betas]


def func_mean(args, r, scaling):
    return [math.log(r) + math.log(v) - v / scaling for v in args]


def func_mean_generalized(arg, r, theta):
    return math.log(r) + math.log(arg) - (arg**theta)


def func_sigma(dim, sigma):
    return [sigma for _ in range(dim)]


def func_lam(args, phi):
    return [phi * v for v in args]


def func_shape(args, observation):
    return [v + observation for v in args]


def func_scale(beta, phi):
    return [b / (b * phi + 1.0) for b in beta]
