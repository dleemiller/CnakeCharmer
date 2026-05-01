"""User-user Pearson similarity matrix construction."""

from __future__ import annotations

import numpy as np


def get_ratings_for_both_users(active_user_vector, other_user_vector):
    x = []
    y = []
    for i in range(active_user_vector.size):
        if active_user_vector[i] != 0 and other_user_vector[i] != 0:
            x.append(active_user_vector[i])
            y.append(other_user_vector[i])
    return np.asarray(x), np.asarray(y)


def sum_of_squares(arr):
    return float(np.sum(arr * arr))


def calculate_pearson_correlation(movie_user_rating_matrix, active_user_index, user_index):
    x, y = get_ratings_for_both_users(
        movie_user_rating_matrix[:, active_user_index], movie_user_rating_matrix[:, user_index]
    )
    if x.size == 0:
        return 0.0

    mx = x.mean()
    my = y.mean()
    xm = x - mx
    ym = y - my

    r_num = float(np.sum(xm * ym))
    r_den = np.sqrt(sum_of_squares(xm) * sum_of_squares(ym))
    if r_den == 0:
        return 0.0
    return r_num / r_den


def build_weight_matrix_between_users(movie_user_rating_matrix, number_of_users):
    user_weight_matrix = np.zeros((number_of_users, number_of_users), dtype=float)
    for active_user_index in range(number_of_users - 1):
        for other_user_index in range(active_user_index + 1, number_of_users):
            w = calculate_pearson_correlation(
                movie_user_rating_matrix, active_user_index, other_user_index
            )
            user_weight_matrix[active_user_index, other_user_index] = w
            user_weight_matrix[other_user_index, active_user_index] = w
    return user_weight_matrix
