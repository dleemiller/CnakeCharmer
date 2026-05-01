"""Simple topic model with collapsed Gibbs sampling updates."""

from __future__ import annotations

import numpy as np


class Document:
    def __init__(self, doc_tokens, doc_topics, topic_changes, doc_topic_counts):
        self.doc_tokens = doc_tokens
        self.doc_topics = doc_topics
        self.topic_changes = topic_changes
        self.doc_topic_counts = doc_topic_counts


class TopicModel:
    def __init__(self, num_topics, vocabulary, doc_smoothing, word_smoothing):
        self.num_topics = num_topics
        self.vocabulary = list(vocabulary)
        self.vocab_size = len(vocabulary)
        self.doc_smoothing = doc_smoothing
        self.word_smoothing = word_smoothing
        self.smoothing_times_vocab_size = word_smoothing * self.vocab_size
        self.topic_totals = np.zeros(num_topics, dtype=int)
        self.word_topics = np.zeros((self.vocab_size, num_topics), dtype=int)
        self.documents = []

    def add_document(self, doc):
        self.documents.append(doc)
        for i in range(len(doc.doc_tokens)):
            word_id = doc.doc_tokens[i]
            topic = doc.doc_topics[i]
            self.word_topics[word_id, topic] += 1
            self.topic_totals[topic] += 1
            doc.doc_topic_counts[topic] += 1

    def sample(self, iterations):
        topic_normalizers = np.zeros(self.num_topics, dtype=float)
        for topic in range(self.num_topics):
            topic_normalizers[topic] = 1.0 / (
                self.topic_totals[topic] + self.smoothing_times_vocab_size
            )

        for _ in range(iterations):
            for document in self.documents:
                doc_tokens = document.doc_tokens
                doc_topics = document.doc_topics
                doc_topic_counts = document.doc_topic_counts
                doc_length = len(doc_tokens)
                uniform_variates = np.random.random_sample(doc_length)

                for i in range(doc_length):
                    word_id = doc_tokens[i]
                    old_topic = doc_topics[i]
                    word_topic_counts = self.word_topics[word_id, :]

                    word_topic_counts[old_topic] -= 1
                    self.topic_totals[old_topic] -= 1
                    doc_topic_counts[old_topic] -= 1

                    topic_probs = np.zeros(self.num_topics, dtype=float)
                    sampling_sum = 0.0
                    for topic in range(self.num_topics):
                        p = (
                            (doc_topic_counts[topic] + self.doc_smoothing)
                            * (word_topic_counts[topic] + self.word_smoothing)
                            / (self.topic_totals[topic] + self.smoothing_times_vocab_size)
                        )
                        sampling_sum += p
                        topic_probs[topic] = sampling_sum

                    sample = uniform_variates[i] * sampling_sum
                    new_topic = int(np.searchsorted(topic_probs, sample, side="left"))
                    if new_topic >= self.num_topics:
                        new_topic = self.num_topics - 1

                    doc_topics[i] = new_topic
                    word_topic_counts[new_topic] += 1
                    self.topic_totals[new_topic] += 1
                    doc_topic_counts[new_topic] += 1
