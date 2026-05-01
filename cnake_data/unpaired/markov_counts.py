"""Count state transitions from chunked trajectories."""

from __future__ import annotations

import numpy as np


def update_counts(counts, states, prev):
    counts[prev, states[0]] += 1
    for i in range(states.shape[0] - 1):
        counts[states[i], states[i + 1]] += 1
    return states[states.shape[0] - 1]


def update_counts_first(counts, states):
    for i in range(states.shape[0] - 1):
        counts[states[i], states[i + 1]] += 1
    return states[states.shape[0] - 1]


def make_counts(chunkedtrajs, n_states, dref):
    counts = np.zeros((n_states, n_states), dtype=np.int32)
    for chunks in chunkedtrajs:
        fc = np.asarray(dref(chunks[0]), dtype=np.int32)
        fc.flags["WRITEABLE"] = True
        last = update_counts_first(counts, fc)
        for c in chunks[1:]:
            c2 = np.asarray(dref(c), dtype=np.int32)
            c2.flags["WRITEABLE"] = True
            last = update_counts(counts, c2, last)
    return np.asarray(counts)
