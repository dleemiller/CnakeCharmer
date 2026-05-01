"""Group tree children by depth and track max depth."""

from __future__ import annotations


def get_depth_sorted_children(children):
    groups = {}
    max_depth = 0
    for child in children:
        d = child.depth
        groups.setdefault(d, []).append(child)
        if d > max_depth:
            max_depth = d
    groups[-1] = max_depth
    return groups


def depth_sorted_children(tree):
    return get_depth_sorted_children(tree.general_children)
