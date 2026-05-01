"""Distance based on stationary character distribution from generated contexts."""

from __future__ import annotations


class StationaryDistribution:
    def distance(self, left_vlmc, right_vlmc):
        left_prob = self._find_stationary_probability(left_vlmc)
        right_prob = self._find_stationary_probability(right_vlmc)
        return sum(abs(left_prob[ch] - right_prob[ch]) for ch in left_vlmc.alphabet)

    def _find_stationary_probability(self, vlmc):
        sequence_length = 2000
        state_count = self._count_state_occurrences(vlmc, sequence_length)
        char_probs = {}
        for ch in vlmc.alphabet:
            prob = 0.0
            for key, value in state_count.items():
                prob_state = value / sequence_length
                prob_char = vlmc.tree[key][ch]
                prob += prob_char * prob_state
            char_probs[ch] = prob
        return char_probs

    def _count_state_occurrences(self, vlmc, sequence_length):
        sequence = vlmc.generate_sequence(sequence_length, 500)
        state_count = {}
        for i in range(sequence_length):
            current = sequence[0:i][-vlmc.order :]
            state = vlmc.get_context(current)
            state_count[state] = state_count.get(state, 0) + 1
        return state_count
