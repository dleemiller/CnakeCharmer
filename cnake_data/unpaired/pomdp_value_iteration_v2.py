"""Policy-tree based finite-horizon value iteration."""

from __future__ import annotations


class PolicyTreeNode:
    def __init__(self, action, depth, agent, discount_factor, children=None):
        self.action = action
        self.depth = depth
        self._agent = agent
        self.children = {} if children is None else children
        self._discount_factor = discount_factor
        self.values = self._compute_values()

    def _compute_values(self):
        observations = self._agent.all_observations
        states = self._agent.all_states
        discount_factor = self._discount_factor**self.depth
        values = {}
        for s in states:
            expected_future_value = 0.0
            for sp in states:
                for o in observations:
                    trans_prob = self._agent.transition_model.probability(sp, s, self.action)
                    obsrv_prob = self._agent.observation_model.probability(o, sp, self.action)
                    subtree_value = self.children[o].values[s] if len(self.children) > 0 else 0.0
                    reward = self._agent.reward_model.sample(s, self.action, sp)
                    expected_future_value += (
                        trans_prob * obsrv_prob * (reward + discount_factor * subtree_value)
                    )
            values[s] = expected_future_value
        return values


class ValueIteration:
    def __init__(self, horizon=1, discount_factor=0.9, epsilon=1e-6):
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._planning_horizon = horizon

    def plan(self, agent, policy_trees):
        value_beliefs = {}
        for p, policy_tree in enumerate(policy_trees):
            value_beliefs[p] = 0.0
            for state in agent.all_states:
                value_beliefs[p] += agent.cur_belief[state] * policy_tree.values[state]
        pmax = max(value_beliefs, key=value_beliefs.get)
        return policy_trees[pmax].action
