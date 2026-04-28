import random


def initialize_alias_tables(weights):
    """Build alias and probability tables for O(1) sampling."""
    k = len(weights)
    scaled = [w * k for w in weights]
    prob = [0.0] * k
    alias = [0] * k

    small = [i for i, p in enumerate(scaled) if p < 1.0]
    large = [i for i, p in enumerate(scaled) if p >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in small + large:
        prob[i] = 1.0

    return prob, alias


def generate_one(prob, alias):
    k = len(prob)
    i = random.randrange(k)
    if random.random() <= prob[i]:
        return i
    return alias[i]


def generate_many(n, prob, alias):
    return [generate_one(prob, alias) for _ in range(n)]


def gen_samples_alias(n, weights):
    prob, alias = initialize_alias_tables(weights)
    return generate_many(n, prob, alias)
