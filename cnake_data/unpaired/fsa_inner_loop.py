"""Finite-state approximate-recognition inner loop."""

from __future__ import annotations


def final(n, state, index, word_len):
    j = 0
    is_final = False
    while j < len(state) and not is_final:
        i = state[j].i + index
        e = state[j].e
        if word_len - i + e <= n:
            is_final = True
        j += 1
    return int(is_final)


def inner_loop(states, word, n, transitions_states, fsa_states, fsa_final_states):
    uword = str(word)
    words = []
    window = 2 * n + 1
    word_len = len(uword)

    while states:
        all_state = states.pop()
        v, q, state_type, index = all_state

        chunk_size = min(window, word_len - index)
        word_chunk = uword[index : index + chunk_size]

        word_states = transitions_states[chunk_size][repr(state_type)]
        transitions = fsa_states[q].transitions.items()

        for x, q1 in transitions:
            if x in word_chunk:
                cv = 0
                for each in word_chunk:
                    match = each == x
                    cv = (cv << 1) | int(match)
            else:
                cv = 0

            cv = (cv << 1) | chunk_size
            state = word_states[cv]
            stype, new_index = state

            if stype:
                v1 = v + x
                states.append((v1, q1, stype, new_index + index))

        if q in fsa_final_states and final(n, state_type, index, word_len):
            words.append(v)

    return words
