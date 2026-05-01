"""AABB-based quadtree with insert/update/query support."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float


@dataclass
class AABB:
    center: Vector2
    hwidth: float
    hheight: float | None = None

    def __post_init__(self) -> None:
        if self.hheight is None:
            self.hheight = self.hwidth

    def left(self) -> float:
        return self.center.x - self.hwidth

    def right(self) -> float:
        return self.center.x + self.hwidth

    def bottom(self) -> float:
        return self.center.y - self.hheight

    def top(self) -> float:
        return self.center.y + self.hheight

    def intersects(self, other: AABB) -> bool:
        return not (
            self.right() < other.left()
            or self.left() > other.right()
            or self.top() < other.bottom()
            or self.bottom() > other.top()
        )

    def contains_aabb(self, other: AABB) -> bool:
        return (
            self.left() <= other.left()
            and self.right() >= other.right()
            and self.bottom() <= other.bottom()
            and self.top() >= other.top()
        )


@dataclass
class NodeItem:
    aabb: AABB
    flags: int = 0
    ent: object | None = None
    node: Node | None = None

    def update_quadtree(self) -> None:
        if self.node is None:
            return
        node = self.node
        while not node.box.contains_aabb(self.aabb):
            if node.parent is None:
                node.expand()
            else:
                node = node.parent
        if node is not self.node:
            self.node.remove(self)
            node.insert(self)


@dataclass
class Node:
    box: AABB
    depth: int
    parent: Node | None
    items: set[NodeItem] = field(default_factory=set)
    node0: Node | None = None
    node1: Node | None = None
    node2: Node | None = None
    node3: Node | None = None

    def aabb_subnode(self, aabb: AABB) -> Node | None:
        if aabb.hwidth >= self.box.hwidth or aabb.hheight >= self.box.hheight:
            return None
        cx, cy = self.box.center.x, self.box.center.y
        if aabb.bottom() <= cy:
            if aabb.right() <= cx:
                return self.node0
            if aabb.left() >= cx:
                return self.node1
            return None
        if aabb.top() >= cy:
            if aabb.right() <= cx:
                return self.node2
            if aabb.left() >= cx:
                return self.node3
        return None

    def subdivide(self, n: int = 1) -> None:
        qw = self.box.hwidth / 2.0
        qh = self.box.hheight / 2.0
        nd = self.depth + 1
        cx, cy = self.box.center.x, self.box.center.y
        self.node0 = Node(AABB(Vector2(cx - qw, cy - qh), qw, qh), nd, self)
        self.node1 = Node(AABB(Vector2(cx + qw, cy - qh), qw, qh), nd, self)
        self.node2 = Node(AABB(Vector2(cx - qw, cy + qh), qw, qh), nd, self)
        self.node3 = Node(AABB(Vector2(cx + qw, cy + qh), qw, qh), nd, self)

        old_items = self.items
        self.items = set()
        if n > 1:
            self.node0.subdivide(n - 1)
            self.node1.subdivide(n - 1)
            self.node2.subdivide(n - 1)
            self.node3.subdivide(n - 1)
        for item in old_items:
            self.insert(item)

    def expand(self) -> None:
        assert self.parent is None
        self.box.hwidth *= 2.0
        self.box.hheight *= 2.0
        if self.node0 is None:
            return
        prev = (self.node0, self.node1, self.node2, self.node3)
        self.subdivide(2)
        self.node0.node3 = prev[0]
        self.node1.node2 = prev[1]
        self.node2.node1 = prev[2]
        self.node3.node0 = prev[3]
        prev[0].parent = self.node0
        prev[1].parent = self.node1
        prev[2].parent = self.node2
        prev[3].parent = self.node3

    def insert(self, item: NodeItem) -> None:
        if self.node0 is not None:
            subnode = self.aabb_subnode(item.aabb)
            if subnode is not None:
                subnode.insert(item)
                return
        self.items.add(item)
        item.node = self
        if self.node0 is None and len(self.items) >= 5 and self.box.hwidth >= 2.0:
            self.subdivide()

    def remove(self, item: NodeItem) -> None:
        self.items.remove(item)
        item.node = None
        if self.parent is not None and not self.items:
            self.parent.check_collapse()

    def check_collapse(self) -> None:
        if self.node0 is None:
            return
        if any(n.node0 is not None for n in (self.node0, self.node1, self.node2, self.node3)):
            return
        items = self.node0.items | self.node1.items | self.node2.items | self.node3.items
        if len(items) < 5:
            self.items = items
            for item in items:
                item.node = self
            self.node0 = self.node1 = self.node2 = self.node3 = None

    def query_aabb(self, aabb: AABB, out: set[NodeItem], flags: int) -> None:
        if not self.box.intersects(aabb):
            return
        for item in self.items:
            if (item.flags & flags) == flags and aabb.intersects(item.aabb):
                out.add(item)
        if self.node0 is not None:
            self.node0.query_aabb(aabb, out, flags)
            self.node1.query_aabb(aabb, out, flags)
            self.node2.query_aabb(aabb, out, flags)
            self.node3.query_aabb(aabb, out, flags)


class Quadtree:
    def __init__(self) -> None:
        self.root = Node(AABB(Vector2(0.0, 0.0), 65536.0), 0, None)

    def insert(self, item: NodeItem) -> None:
        self.root.insert(item)

    def query_aabb_ents(
        self, aabb: AABB, exclude: set | None, flags: int, components: set | None
    ) -> set:
        result_items: set[NodeItem] = set()
        self.root.query_aabb(aabb, result_items, flags)
        result = {item.ent for item in result_items if item.ent is not None}
        if exclude:
            result -= exclude
        if components:
            return {
                e
                for e in result
                if getattr(e, "_component_names", set()) & components == components
            }
        return result
