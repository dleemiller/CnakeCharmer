import math


def log_add(x, y):
    return x + math.log(1.0 + math.exp(y - x)) if x >= y else y + math.log(1.0 + math.exp(x - y))


def sigma_weak_lensing(z, dl):
    return 0.066 * dl * ((1.0 - (1.0 + z) ** (-0.25)) / 0.25) ** 1.8


def em_selection_function(dl):
    return (1.0 - dl / 12000.0) / (1.0 + (dl / 3700.0) ** 7) ** 1.35


def log_likelihood_single_event(hosts, meandl, sigma, omega, event_redshift, em_selection=False):
    """Single-event GW host-marginalized log likelihood."""
    log_two_pi_by_two = 0.5 * math.log(2.0 * math.pi)
    dl = omega.luminosity_distance(event_redshift)

    logp_detection = 0.0
    logp_nondetection = 0.0
    if em_selection:
        p_det = max(em_selection_function(dl), 1e-300)
        logp_detection = math.log(p_det)
        logp_nondetection = math.log(max(1.0 - p_det, 1e-300))

    weak_err = sigma_weak_lensing(event_redshift, dl)

    logL = -math.inf
    for z_host, zerr_host, ang_w in hosts:
        sigma_z = zerr_host * (1.0 + z_host)
        score_z = (event_redshift - z_host) / sigma_z
        logL_g = -0.5 * score_z * score_z + math.log(ang_w) - math.log(sigma_z) - log_two_pi_by_two
        logL = log_add(logL, logL_g)

    if em_selection:
        logL += logp_detection

    logLn = logp_nondetection if em_selection else -math.inf

    sigma2 = sigma * sigma + weak_err * weak_err
    log_sigma_by_two = 0.5 * math.log(sigma2)
    log_norm = math.log(omega.integrate_comoving_volume_density())
    logP = math.log(omega.uniform_comoving_volume_density(event_redshift)) - log_norm

    return (
        (-0.5 * (dl - meandl) * (dl - meandl) / sigma2 - log_two_pi_by_two - log_sigma_by_two)
        + log_add(logL, logLn)
        + logP
    )
