"""Dummy-variable matrix generation for epistasis interaction models."""

from __future__ import annotations

import numpy as np


def generate_dv_matrix(sequences, interactions, model_type="local"):
    model_options = {"local": {"0": 0, "1": 1}, "global": {"0": -1, "1": 1}}
    encoding = model_options[model_type]

    dim1 = len(sequences)
    dim2 = len(interactions)
    x = np.ones((dim1, dim2), dtype=int)

    for n in range(dim1):
        for i in range(1, dim2):
            element = 1
            for j in range(len(interactions[i])):
                m = interactions[i][j] - 1
                element = element * encoding[sequences[n][m]]
            x[n][i] = element

    return np.asarray(x)
