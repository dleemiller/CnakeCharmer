"""PST/VLMC distance with probability+dissimilarity terms."""

from __future__ import annotations


class PSTMatching:
    def __init__(self, dissimilarity_weight: float):
        self.dissimilarity_weight = dissimilarity_weight

    def distance(self, left_vlmc, right_vlmc):
        union = set(left_vlmc.tree.keys()).union(set(right_vlmc.tree.keys()))
        intersection = set(left_vlmc.tree.keys()).intersection(set(right_vlmc.tree.keys()))
        total = 0.0
        for state in union:
            p_term = (1 - self.dissimilarity_weight) * self.probability_cost(
                state, left_vlmc, right_vlmc
            )
            d_term = self.dissimilarity_weight * self.dissimilarity_cost(
                state, left_vlmc, right_vlmc
            )
            w = self.state_weight(state, left_vlmc, right_vlmc)
            total += w * (p_term + d_term)
        return total / len(intersection) if intersection else 0.0

    def state_weight(self, state, left_vlmc, right_vlmc):
        left_state = left_vlmc.get_context(state)
        right_state = right_vlmc.get_context(state)
        return (
            left_vlmc.occurrence_probability(left_state)
            + right_vlmc.occurrence_probability(right_state)
        ) / 2

    def dissimilarity_cost(self, state, left_vlmc, right_vlmc):
        if state in left_vlmc.tree and state in right_vlmc.tree:
            return 0.0
        if state in left_vlmc.tree:
            return self._dissimilarity_cost(left_vlmc, right_vlmc, state)
        return self._dissimilarity_cost(right_vlmc, left_vlmc, state)

    def _dissimilarity_cost(self, vlmc, vlmc_without_state, state):
        closest = vlmc_without_state.get_context(state)
        diff = abs(len(closest) - len(state))
        max_len = max(len(closest), len(state))
        return diff / max_len if max_len else 0.0

    def probability_cost(self, state, left_vlmc, right_vlmc):
        if state in left_vlmc.tree and state in right_vlmc.tree:
            diff = sum(
                abs(left_vlmc.tree[state][ch] - right_vlmc.tree[state][ch])
                for ch in left_vlmc.alphabet
            )
            return diff / 2
        return 0.0
