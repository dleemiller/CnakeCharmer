import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def score_sij(s_ab, msa_test, pmi, middle_index=None):
    if middle_index is None:
        middle_index = int(msa_test.shape[1] / 2)

    for a in range(s_ab.shape[0]):
        for b in range(s_ab.shape[1]):
            s_ab[a, b] = 0.0
            for i in range(middle_index):
                for j in range(middle_index, msa_test.shape[1]):
                    state_1 = int((msa_test[a, i] + 1) / 2)
                    state_2 = int((msa_test[b, j] + 1) / 2)
                    s_ab[a, b] += pmi[i, j, state_1, state_2]
    return s_ab


def two_body_freq(fij, msa, pseudocount, middle_index):
    delta_freq = 1.0 / msa.shape[0]
    for site_i in range(middle_index):
        for site_j in range(middle_index, msa.shape[1]):
            for row in range(msa.shape[0]):
                state_1 = int((msa[row, site_i] + 1) / 2)
                state_2 = int((msa[row, site_j] + 1) / 2)
                fij[site_i, site_j, state_1, state_2] += delta_freq

            for s1 in range(2):
                for s2 in range(2):
                    fij[site_i, site_j, s1, s2] = (
                        pseudocount / 4 + (1 - pseudocount) * fij[site_i, site_j, s1, s2]
                    )


def pmi_matrix(msa, pseudocount, middle_index=None):
    if middle_index is None:
        middle_index = int(msa.shape[1] / 2)

    pmi = np.zeros((msa.shape[1], msa.shape[1], 2, 2), dtype=float)
    fij = np.zeros((msa.shape[1], msa.shape[1], 2, 2), dtype=float)
    f1 = np.zeros((msa.shape[1], 2), dtype=float)

    two_body_freq(fij, msa, pseudocount, middle_index)

    for site_i in range(msa.shape[1]):
        for row in range(msa.shape[0]):
            s1 = int((msa[row, site_i] + 1) / 2)
            f1[site_i, s1] += 1

        f1[site_i, 0] = pseudocount / 2 + (1 - pseudocount) * f1[site_i, 0] / msa.shape[0]
        f1[site_i, 1] = pseudocount / 2 + (1 - pseudocount) * f1[site_i, 1] / msa.shape[0]

    for site_i in range(middle_index):
        for site_j in range(middle_index, msa.shape[1]):
            for s1 in range(2):
                for s2 in range(2):
                    pmi[site_i, site_j, s1, s2] = math.log(
                        fij[site_i, site_j, s1, s2] / (f1[site_i, s1] * f1[site_j, s2])
                    )

    return pmi


def inference_partner_mutual_info(l_msa, s_train, reg, n_pair, fast=False, middle_index=None):
    percentage_true_partner = np.zeros(n_pair, dtype=float)
    counter_average = 0

    for index_test in range(l_msa.shape[0]):
        l_perm = np.random.permutation(l_msa.shape[1])
        msa_train = l_msa[index_test, l_perm[:s_train]]

        if fast:
            index_max = min(s_train + 50 * n_pair, np.size(l_msa, axis=1))
            msa_test = l_msa[index_test, l_perm[s_train:index_max]]
        else:
            msa_test = l_msa[index_test, l_perm[s_train:]]

        pmi = pmi_matrix(msa_train, reg, middle_index)
        ind_species = n_pair
        cost = np.zeros((n_pair, n_pair), dtype=float)

        while ind_species < msa_test.shape[0]:
            cost = -1.0 * score_sij(
                cost, msa_test[ind_species - n_pair : ind_species], pmi, middle_index
            )
            perm_rows = np.random.permutation(cost.shape[0])
            cost = cost[perm_rows]
            row_ind, col_ind = linear_sum_assignment(cost)
            percentage_true_partner += perm_rows[row_ind] == col_ind
            counter_average += 1
            ind_species += n_pair

    return float(np.mean(percentage_true_partner / counter_average))
