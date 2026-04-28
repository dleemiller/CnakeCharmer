"""Simple labeled tree node utilities."""

from __future__ import annotations

import os


class TreeItem:
    def __init__(self, label=None, parent=None):
        self.children = []
        self.parent = parent
        self.label = label
        if parent:
            parent.add_child(self)
            self.order = parent.order + 1
        else:
            self.order = 0
        self.extra = None

    def add_child(self, child: TreeItem):
        self.children.append(child)

    def has_child(self, label: str) -> bool:
        return label in [i.label for i in self.children]

    def get(self, label: str):
        for i in self.children:
            if i.label == label:
                return i
        return None

    def sort(self):
        for child in self.children:
            child.sort()
        self.children.sort(key=lambda x: x.label)

    def next(self):
        if self.parent is None:
            raise IndexError("Next does not exist")
        idx = self.parent.children.index(self)
        if idx < len(self.parent.children) - 1:
            return self.parent.children[idx + 1]
        return self.parent.next().children[0]

    def previous(self):
        if self.parent is None:
            raise IndexError("Previous does not exist")
        idx = self.parent.children.index(self)
        if idx > 0:
            return self.parent.children[idx - 1]
        return self.parent.previous().children[-1]

    def first(self):
        return self.children[0].first() if self.children else self

    def last(self):
        return self.children[-1].last() if self.children else self

    @property
    def size(self):
        if self.children:
            return sum(child.size for child in self.children)
        return 1

    @property
    def name(self):
        if self.order <= 1:
            return self.label or ""
        if self.order == 4:
            return self.parent.name + os.sep + self.label
        return self.parent.name + "-" + self.label
