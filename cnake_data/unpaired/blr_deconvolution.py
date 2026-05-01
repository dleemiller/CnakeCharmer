"""Accumulator-based baseline restoration deconvolution."""

from __future__ import annotations

import math


def deconvolve_signal(
    signal_daq,
    n_baseline=28000,
    coef_blr=1.632411e-03,
    thr_trigger=5.0,
    acum_discharge_length=5000,
):
    coef = coef_blr
    thr_acum = thr_trigger / coef
    n = len(signal_daq)
    signal_r = [0.0] * n
    acum = [0.0] * n

    baseline = sum(signal_daq[:n_baseline]) / float(n_baseline)
    sig = [baseline - x for x in signal_daq]

    nn = min(400, n)
    noise = sum(v * v for v in sig[:nn]) / float(nn)
    trigger_line = thr_trigger * math.sqrt(noise)

    j = 0
    signal_r[0] = sig[0]
    for k in range(1, n):
        signal_r[k] = sig[k] + sig[k] * (coef / 2.0) + coef * acum[k - 1]
        acum[k] = acum[k - 1] + sig[k]
        if (sig[k] < trigger_line) and (acum[k - 1] < thr_acum):
            if acum[k - 1] > 1.0:
                acum[k] = acum[k - 1] * (1.0 - coef)
                j = min(j + 1, acum_discharge_length - 1)
            else:
                acum[k] = 0.0
                j = 0
    return signal_r
