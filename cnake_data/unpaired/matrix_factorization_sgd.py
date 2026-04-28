import random


def fit_matrix_factorization_sgd(
    coo_rows,
    coo_cols,
    coo_data,
    n_users,
    n_items,
    epochs,
    steps_per_epoch,
    num_factors,
    learning_rate,
    regularization,
    seed=0,
):
    """Train user/item factors with sampled SGD on COO interactions."""
    rng = random.Random(seed)
    n_interactions = len(coo_data)

    user_factors = [[rng.random() for _ in range(num_factors)] for _ in range(n_users)]
    item_factors = [[rng.random() for _ in range(num_factors)] for _ in range(n_items)]

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            sample_index = rng.randrange(n_interactions)
            user_id = coo_rows[sample_index]
            item_id = coo_cols[sample_index]
            rating = coo_data[sample_index]

            predicted = 0.0
            for f in range(num_factors):
                predicted += user_factors[user_id][f] * item_factors[item_id][f]

            err = rating - predicted

            for f in range(num_factors):
                h_i = item_factors[item_id][f]
                w_u = user_factors[user_id][f]

                user_update = err * h_i - regularization * w_u
                item_update = err * w_u - regularization * h_i

                user_factors[user_id][f] += learning_rate * user_update
                item_factors[item_id][f] += learning_rate * item_update

    return user_factors, item_factors
