import math

import numpy as np


def efun(z):
    return -5e2 if z < -5e2 else z


def gate_inf(x, a, b):
    return 1.0 / (1.0 + math.exp(-a * x + b))


def tau_gate2(x, c, d, e, f, g, h):
    y = x - c
    return d / (math.exp(-(e * y + f * y**2)) + math.exp(g * y + h * y**2))


def tau_gate3(x, c, d, e, f, g, h, k, l):
    y = x - c
    return d / (math.exp(-(e * y + f * y**2 + k * y**3)) + math.exp(g * y + h * y**2 + l * y**3))


def seed(n):
    np.random.seed(n)


def updatexpe2(v, act_var, i, tstep, a_act, b_act, c_act, d_act, e_act, f_act, g_act, h_act, tadj):
    len_act_var = len(v[:, 0])
    for j in range(len_act_var):
        act_var_inf1 = gate_inf(v[j, i - 1], a_act, b_act)
        act_var[j, i] = act_var_inf1 + (act_var[j, i - 1] - act_var_inf1) * math.exp(
            efun(-tstep * tadj / tau_gate2(v[j, i - 1], c_act, d_act, e_act, f_act, g_act, h_act))
        )


def expeuler2(
    t,
    v,
    act_var,
    i_channel,
    tstep,
    p_act,
    a_act,
    b_act,
    c_act,
    d_act,
    e_act,
    f_act,
    g_act,
    h_act,
    e_channel,
    fact_inward,
):
    celsius = 37
    temp = 23
    q10 = 2.3
    tadj = q10 ** ((celsius - temp) / 10)
    gbar_channel = 1.0

    len_act_var = len(v[:, 0])
    for j in range(len_act_var):
        act_var[j, 0] = gate_inf(v[j, 0], a_act, b_act)

    for i in range(1, t.shape[0]):
        updatexpe2(
            v, act_var, i, tstep, a_act, b_act, c_act, d_act, e_act, f_act, g_act, h_act, tadj
        )

    i_channel = fact_inward * tadj * gbar_channel * (act_var**p_act) * (v - e_channel)
    i_channel /= np.max(i_channel)
    return i_channel


def updatexpe3(
    v, act_var, i, tstep, a_act, b_act, c_act, d_act, e_act, f_act, g_act, h_act, k_act, l_act, tadj
):
    len_act_var = len(v[:, 0])
    for j in range(len_act_var):
        act_var_inf1 = gate_inf(v[j, i - 1], a_act, b_act)
        act_var[j, i] = act_var_inf1 + (act_var[j, i - 1] - act_var_inf1) * math.exp(
            efun(
                -tstep
                * tadj
                / tau_gate3(v[j, i - 1], c_act, d_act, e_act, f_act, g_act, h_act, k_act, l_act)
            )
        )


def expeuler3(
    t,
    v,
    act_var,
    i_channel,
    tstep,
    p_act,
    a_act,
    b_act,
    c_act,
    d_act,
    e_act,
    f_act,
    g_act,
    h_act,
    k_act,
    l_act,
    e_channel,
    fact_inward,
):
    celsius = 37
    temp = 23
    q10 = 2.3
    tadj = q10 ** ((celsius - temp) / 10)
    gbar_channel = 1.0

    len_act_var = len(v[:, 0])
    for j in range(len_act_var):
        act_var[j, 0] = gate_inf(v[j, 0], a_act, b_act)

    for i in range(1, t.shape[0]):
        updatexpe3(
            v,
            act_var,
            i,
            tstep,
            a_act,
            b_act,
            c_act,
            d_act,
            e_act,
            f_act,
            g_act,
            h_act,
            k_act,
            l_act,
            tadj,
        )

    i_channel = fact_inward * tadj * gbar_channel * (act_var**p_act) * (v - e_channel)
    i_channel /= np.max(i_channel)
    return i_channel
