from math import exp, log


def network_log_likelihood(n):
    """Compute the log-likelihood of a multi-layer network model.

    Creates a synthetic network with n nodes, 2 layers, and 3 latent features.
    The model computes:
        loglik = sum over (k, i, j) of [ Y[k,i,j] * eta - log(1 + exp(eta)) ]
    where eta = delta[k,i] + delta[k,j] + sum_p(lambda[k,p] * X[i,p] * X[j,p])

    This is a logistic network likelihood used in latent space models
    for dynamic networks.

    Args:
        n: Number of nodes in the network.

    Returns:
        (log_likelihood, num_edges, num_evaluated_pairs) rounded to 8 decimals.
    """
    n_layers = 2
    n_features = 3

    # Build deterministic adjacency (upper triangle, -1 = missing)
    Y = [[[0.0] * n for _ in range(n)] for _ in range(n_layers)]
    for k in range(n_layers):
        for i in range(n):
            for j in range(i):
                val = (i * 11 + j * 7 + k * 3) % 10
                if val < 3:
                    Y[k][i][j] = 1.0
                elif val < 5:
                    Y[k][i][j] = -1.0  # missing
                else:
                    Y[k][i][j] = 0.0

    # Build latent positions X: n x n_features
    X = [[0.0] * n_features for _ in range(n)]
    for i in range(n):
        for p in range(n_features):
            X[i][p] = ((i * 13 + p * 29) % 100) / 100.0 - 0.5

    # Build layer-specific parameters
    lmbda = [[0.0] * n_features for _ in range(n_layers)]
    delta = [[0.0] * n for _ in range(n_layers)]
    for k in range(n_layers):
        for p in range(n_features):
            lmbda[k][p] = ((k * 37 + p * 19) % 100) / 100.0 - 0.5
        for i in range(n):
            delta[k][i] = ((k * 41 + i * 23) % 100) / 100.0 - 0.5

    # Compute log-likelihood
    loglik = 0.0
    num_edges = 0
    num_evaluated = 0

    for k in range(n_layers):
        for i in range(n):
            for j in range(i):
                if Y[k][i][j] != -1.0:
                    eta = delta[k][i] + delta[k][j]
                    for p in range(n_features):
                        eta += lmbda[k][p] * X[i][p] * X[j][p]
                    loglik += Y[k][i][j] * eta - log(1 + exp(eta))
                    num_evaluated += 1
                    if Y[k][i][j] == 1.0:
                        num_edges += 1

    return (round(loglik, 8), num_edges, num_evaluated)
