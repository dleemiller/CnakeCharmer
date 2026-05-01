"""Recurrent linear policy scoring with action-history coupling."""

from __future__ import annotations

import numpy as np


def calc_reg_for_action(
    action: int,
    state: np.ndarray,
    c: np.ndarray,
    cc: np.ndarray,
    ac: np.ndarray,
    fc: np.ndarray,
    last_actions: np.ndarray,
) -> float:
    s = float(np.sum(cc[action] * state * state + c[action] * state))
    s += float(np.sum(last_actions * ac[action]))
    return s + float(fc[action])


def get_action_by_state_fast(
    state: np.ndarray,
    c: np.ndarray,
    cc: np.ndarray,
    ac: np.ndarray,
    fc: np.ndarray,
    last_actions: np.ndarray,
) -> int:
    n_actions = c.shape[0]
    best_act = -1
    best_val = -1e9
    for act in range(n_actions):
        val = calc_reg_for_action(act, state, c, cc, ac, fc, last_actions)
        if val > best_val:
            best_val = val
            best_act = act
    last_actions[:] = 0.0
    last_actions[best_act] = 1.0
    return best_act
