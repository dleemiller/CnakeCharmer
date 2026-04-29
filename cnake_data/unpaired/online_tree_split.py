"""Online tree split selection and recursive update helpers."""

from __future__ import annotations

import random


def bin_split(sample_feature, feature_value):
    left = [x[1] for x in sample_feature if x[0] <= feature_value]
    right = [x[1] for x in sample_feature if x[0] > feature_value]
    return left, right


class OnlineTree:
    def __init__(self, number_of_features, number_of_functions=10, min_sample_split=20):
        self.number_of_features = number_of_features
        self.number_of_functions = number_of_functions
        self.min_sample_split = min_sample_split
        self.max_sample = 100
        self.left = None
        self.right = None
        self.criterion = None
        self.randomly_selected_features = []
        self.samples = {}
        self.y = []
        self._randomly_select()

    def _randomly_select(self):
        if self.number_of_features < self.number_of_functions:
            raise ValueError("The feature number is more than maximum")
        feats = set()
        while len(feats) < self.number_of_functions:
            feats.add(random.randint(0, self.number_of_features - 1))
        self.randomly_selected_features = list(feats)
        self.samples = {feature: [] for feature in self.randomly_selected_features}

    def is_leaf(self):
        return self.criterion is None

    def update(self, x, target):
        n = len(self.y)
        if self.is_leaf():
            if n <= self.max_sample:
                self._update_samples(x, target)
            if n == self.min_sample_split or n == 2 * self.min_sample_split:
                self._apply_best_split()
        else:
            feat, threshold = self.criterion
            child = self.left if x[feat] <= threshold else self.right
            child.update(x, target)

    def _update_samples(self, x, target):
        for feat in self.randomly_selected_features:
            self.samples[feat].append((x[feat], target))
        self.y.append(target)

    def _apply_best_split(self):
        best = None
        best_score = float("inf")
        for feat in self.randomly_selected_features:
            sf = sorted(self.samples[feat], key=lambda z: z[0])
            if not sf:
                continue
            pivot = sf[len(sf) // 2][0]
            left, right = bin_split(sf, pivot)
            score = abs(len(left) - len(right))
            if score < best_score:
                best_score = score
                best = (feat, pivot)

        if best is None:
            return

        self.criterion = best
        self.left = OnlineTree(
            self.number_of_features, self.number_of_functions, self.min_sample_split
        )
        self.right = OnlineTree(
            self.number_of_features, self.number_of_functions, self.min_sample_split
        )
