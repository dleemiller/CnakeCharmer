"""Naive Bayes score helpers for Bernoulli and Multinomial models."""

from __future__ import annotations

import math


def multivariate_bernoulli_word_prob(cat, w, words, pwc):
    if w in words:
        return math.log(pwc[cat][w])
    if w in pwc[cat]:
        return math.log(1.0 - pwc[cat][w])
    return 0.0


def multivariate_bernoulli_scores(words, cats, vocab, pc, pwc):
    label_scores = {}
    for cat in cats:
        score = math.log(pc[cat])
        for w in vocab:
            try:
                score += multivariate_bernoulli_word_prob(cat, w, words, pwc)
            except KeyError:
                score = float("-inf")
                break
        label_scores[cat] = score
    return label_scores


def multinomial_scores(cats, cnt, pc, pwc):
    label_scores = {}
    for cat in cats:
        score = math.log(pc[cat])
        for w, c in cnt.items():
            try:
                score += c * math.log(pwc[cat][w])
            except KeyError:
                score = float("-inf")
                break
        label_scores[cat] = score
    return label_scores
