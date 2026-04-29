"""Multimodal LDA Gibbs-update utilities."""

from __future__ import annotations

import math
import random

ALPHA = 1.0
BETA = 1.0


def conv_to_word_list(hist):
    doc = []
    for token_id, count in enumerate(hist):
        for _ in range(int(count)):
            doc.append(token_id)
    return doc


def sample_topic(d, w, n_dz, n_zw, n_z, k_topics, vocab_size):
    probs = [0.0] * k_topics
    for z in range(k_topics):
        probs[z] = (n_dz[d][z] + ALPHA) * (n_zw[z][w] + BETA) / (n_z[z] + vocab_size * BETA)
    for z in range(1, k_topics):
        probs[z] += probs[z - 1]

    r = probs[-1] * random.random()
    for z, p in enumerate(probs):
        if p >= r:
            return z
    return k_topics - 1


def calc_likelihood(data, n_dz, n_zw, n_z, k_topics, vocab_size):
    lik = 0.0
    for d in range(len(data)):
        norm = 1.0 / (sum(n_dz[d]) + k_topics * ALPHA)
        for w in range(vocab_size):
            s = 0.0
            for z in range(k_topics):
                s += (
                    (n_zw[z][w] + BETA) / (n_z[z] + vocab_size * BETA) * (n_dz[d][z] + ALPHA) * norm
                )
            if data[d][w] > 0:
                lik += data[d][w] * math.log(max(s, 1e-12))
    return lik
