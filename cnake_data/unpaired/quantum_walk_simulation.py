"""Simple 1D discrete-time quantum walk simulator."""

from __future__ import annotations

from math import sqrt

H = ((1 / sqrt(2), 1 / sqrt(2)), (1 / sqrt(2), -1 / sqrt(2)))


def apply_coin_operator(spin_states: list[list[complex]]) -> list[list[complex]]:
    out: list[list[complex]] = []
    for up, down in spin_states:
        out_up = H[0][0] * up + H[0][1] * down
        out_down = H[1][0] * up + H[1][1] * down
        out.append([out_up, out_down])
    return out


def apply_shift_operator(
    node_states: list[complex], spin_states: list[list[complex]]
) -> tuple[list[complex], list[list[complex]]]:
    n = len(node_states)
    node_update = [0j] * n
    spin_update = [[0j, 0j] for _ in range(n)]

    for i, amp in enumerate(node_states):
        up, down = spin_states[i]
        if up != 0:
            left = i - 1 if i > 0 else n - 1
            node_update[left] += amp
            spin_update[left][0] += up
        if down != 0:
            right = i + 1 if i < n - 1 else 0
            node_update[right] += amp
            spin_update[right][1] += down

    return node_update, spin_update


def normalize(values: list[complex]) -> list[complex]:
    norm = sqrt(sum(abs(v) ** 2 for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


def normalize_spin(spin_states: list[list[complex]]) -> list[list[complex]]:
    total = sum(abs(v) ** 2 for pair in spin_states for v in pair)
    norm = sqrt(total)
    if norm == 0:
        return spin_states
    return [[pair[0] / norm, pair[1] / norm] for pair in spin_states]


def node_probabilities(node_states: list[complex]) -> list[float]:
    return [float(abs(v) ** 2) for v in node_states]


def quantum_walk(
    number_of_nodes: int, starting_node: int, starting_spin: list[complex], number_of_steps: int
) -> list[float]:
    node_states = [0j] * number_of_nodes
    node_states[starting_node] = 1.0 + 0j
    spin_states = [[0j, 0j] for _ in range(number_of_nodes)]
    spin_states[starting_node] = [starting_spin[0], starting_spin[1]]

    for _ in range(number_of_steps):
        spin_states = apply_coin_operator(spin_states)
        node_states, spin_states = apply_shift_operator(node_states, spin_states)
        node_states = normalize(node_states)
        spin_states = normalize_spin(spin_states)

    return node_probabilities(node_states)
