"""LDA-style topic sequence initialization and count structure setup."""

from __future__ import annotations

import random


def init_seqs_and_counts(num_topics: int, num_terms: int, corpus):
    term_seqs = corpus.term_seqs
    topic_seqs = []
    for di in range(len(term_seqs)):
        topic_seq = [random.randrange(num_topics) for _ in range(len(term_seqs[di]))]
        topic_seqs.append(topic_seq)

    term_topic_counts = [[0] * num_topics for _ in range(num_terms)]
    for di in range(len(term_seqs)):
        for term, topic in zip(term_seqs[di], topic_seqs[di], strict=False):
            term_topic_counts[term][topic] += 1

    terms_per_topic = [0] * num_topics
    for topic in range(num_topics):
        s = 0
        for term in range(num_terms):
            s += term_topic_counts[term][topic]
        terms_per_topic[topic] = s

    return term_seqs, topic_seqs, term_topic_counts, terms_per_topic


def init_priors(num_topics: int, num_terms: int):
    alpha = [1.0 / num_topics for _ in range(num_topics)]
    beta = [1.0 / num_terms for _ in range(num_terms)]
    w_beta = sum(beta)
    return {"alpha": alpha, "beta": beta, "w_beta": w_beta}
