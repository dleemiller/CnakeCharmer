"""SLIM-style MSE training loop for item-item matrix."""

from __future__ import annotations

import random
import time

import numpy as np


def train(urm, learning_rate_input, epochs, n_samples):
    urm_coo = urm.tocoo()
    n_items = urm.shape[1]
    n_interactions = urm.nnz

    urm_coo_row = urm_coo.row
    urm_coo_col = urm_coo.col
    urm_coo_data = urm_coo.data
    urm_indices = urm.indices
    urm_indptr = urm.indptr
    urm_data = urm.data

    item_item_s = np.zeros((n_items, n_items), dtype=float)
    learning_rate = float(learning_rate_input)

    for n_epoch in range(epochs):
        loss = 0.0
        ts = time.time()
        for sample_num in range(n_samples):
            index = random.randrange(n_interactions)

            user_id = urm_coo_row[index]
            item_id = urm_coo_col[index]
            true_rating = urm_coo_data[index]

            start_profile = urm_indptr[user_id]
            end_profile = urm_indptr[user_id + 1]
            predicted_rating = 0.0

            for idx in range(start_profile, end_profile):
                profile_item_id = urm_indices[idx]
                profile_rating = urm_data[idx]
                predicted_rating += item_item_s[profile_item_id, item_id] * profile_rating

            prediction_error = true_rating - predicted_rating
            loss += prediction_error**2

            for idx in range(start_profile, end_profile):
                profile_item_id = urm_indices[idx]
                profile_rating = urm_data[idx]
                item_item_s[profile_item_id, item_id] += (
                    learning_rate * prediction_error * profile_rating
                )

        _ = (n_epoch, loss / max(sample_num, 1), time.time() - ts)

    return item_item_s
