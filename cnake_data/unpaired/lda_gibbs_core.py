"""Collapsed Gibbs sampling core for LDA."""

from __future__ import annotations

import random


def sample_document_topics(
    doc_tokens,
    doc_topics,
    doc_topic_counts,
    word_topics,
    topic_totals,
    topic_probs,
    topic_norms,
    doc_smoothing,
    word_smoothing,
    smooth_vocab,
):
    k_topics = len(topic_totals)
    for i, word_id in enumerate(doc_tokens):
        old_topic = doc_topics[i]

        word_topics[word_id][old_topic] -= 1
        topic_totals[old_topic] -= 1
        doc_topic_counts[old_topic] -= 1
        topic_norms[old_topic] = 1.0 / (topic_totals[old_topic] + smooth_vocab)

        s = 0.0
        for t in range(k_topics):
            p = (
                (doc_topic_counts[t] + doc_smoothing)
                * (word_topics[word_id][t] + word_smoothing)
                * topic_norms[t]
            )
            topic_probs[t] = p
            s += p

        draw = random.random() * s
        new_topic = 0
        while draw > topic_probs[new_topic]:
            draw -= topic_probs[new_topic]
            new_topic += 1

        word_topics[word_id][new_topic] += 1
        topic_totals[new_topic] += 1
        doc_topic_counts[new_topic] += 1
        topic_norms[new_topic] = 1.0 / (topic_totals[new_topic] + smooth_vocab)
        doc_topics[i] = new_topic
