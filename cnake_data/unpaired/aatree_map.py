"""Persistent AA-tree map primitives."""

from __future__ import annotations


class AANode:
    __slots__ = ("level", "l", "r", "key", "val")

    def __init__(self, level, l, r, key, val):
        self.level = level
        self.l = l
        self.r = r
        self.key = key
        self.val = val


TREE_NIL = AANode(0, None, None, None, None)
TREE_NIL.l = TREE_NIL
TREE_NIL.r = TREE_NIL


def skew(n):
    if n.level != 0 and n.l.level == n.level:
        return AANode(
            n.level,
            n.l.l,
            AANode(n.level, n.l.r, n.r, n.key, n.val),
            n.l.key,
            n.l.val,
        )
    return n


def split(n):
    if n.level != 0 and n.r.r.level == n.level:
        return AANode(
            n.r.level + 1,
            AANode(n.level, n.l, n.r.l, n.key, n.val),
            n.r.r,
            n.r.key,
            n.r.val,
        )
    return n


def tree_insert_multi(n, key, val):
    if n.level == 0:
        return AANode(1, TREE_NIL, TREE_NIL, key, val)
    if key < n.key:
        n0 = AANode(n.level, tree_insert_multi(n.l, key, val), n.r, n.key, n.val)
    else:
        n0 = AANode(n.level, n.l, tree_insert_multi(n.r, key, val), n.key, n.val)
    return split(skew(n0))
