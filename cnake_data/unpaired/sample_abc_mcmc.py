"""Sample-based MCMC-ABC sampler."""

from __future__ import annotations

import random

import numpy as np


def sample_abc(
    simu,
    fixed_params: np.ndarray,
    observed: np.ndarray,
    distance,
    distribs,
    n_output: int,
    epsilon: float,
    n_samples: int,
    init_guess: np.ndarray,
    sd: float,
    symmetric_proposal: bool = True,
) -> np.ndarray:
    n_params = len(distribs)
    n_simu = observed.shape[0]

    result = np.empty((n_output, n_params), dtype=float)
    result[0, :] = init_guess

    acc_counter = 0
    for ii in range(1, n_output):
        params_prop = np.empty(n_params, dtype=float)
        for jj in range(n_params):
            params_prop[jj] = distribs[jj].rvs(result[ii - 1, jj], sd)

        simulated = simu.run(params_prop, fixed_params, n_simu)

        accept_mh = True
        if not symmetric_proposal:
            accprob = 1.0
            for jj in range(n_params):
                num = distribs[jj].pdf(result[ii - 1, jj], params_prop[jj], sd)
                den = distribs[jj].pdf(params_prop[jj], result[ii - 1, jj], sd)
                accprob *= num / den
            accept_mh = accprob >= random.random()

        result[ii, :] = result[ii - 1, :]
        if accept_mh:
            accept_samples = True
            for _ in range(n_samples):
                kk = int(random.random() * n_simu)
                if abs(observed[kk] - simulated[kk]) > epsilon:
                    accept_samples = False
                    break
            if accept_samples:
                result[ii, :] = params_prop
                acc_counter += 1

    print(f"Sample-ABC accepted {100.0 * acc_counter / n_output:.3f}% of proposals")
    return result
