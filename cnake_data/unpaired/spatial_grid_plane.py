from contextlib import suppress


def drange(v0, v1, d):
    """Discrete coordinate bins intersecting [v0, v1)."""
    if not (v0 < v1):
        return range(0)
    return range(int(v0) // d, int(v1 + d) // d)


class Plane:
    """Grid-backed spatial index for axis-aligned bbox objects."""

    def __init__(self, bbox, gridsize=80):
        self._seq = []
        self._objs = set()
        self._grid = {}
        self.gridsize = gridsize
        self.x0, self.y0, self.x1, self.y1 = bbox

    def __iter__(self):
        return (obj for obj in self._seq if obj in self._objs)

    def __len__(self):
        return len(self._objs)

    def __contains__(self, obj):
        return obj in self._objs

    def _getrange(self, bbox):
        x0, y0, x1, y1 = bbox
        if x1 <= self.x0 or self.x1 <= x0 or y1 <= self.y0 or self.y1 <= y0:
            return []

        x0 = max(self.x0, x0)
        y0 = max(self.y0, y0)
        x1 = min(self.x1, x1)
        y1 = min(self.y1, y1)

        ret = []
        for y in drange(y0, y1, self.gridsize):
            for x in drange(x0, x1, self.gridsize):
                ret.append((x, y))
        return ret

    def add(self, obj):
        for k in self._getrange((obj.x0, obj.y0, obj.x1, obj.y1)):
            self._grid.setdefault(k, []).append(obj)
        self._seq.append(obj)
        self._objs.add(obj)

    def extend(self, objs):
        for obj in objs:
            self.add(obj)

    def remove(self, obj):
        for k in self._getrange((obj.x0, obj.y0, obj.x1, obj.y1)):
            with suppress(KeyError, ValueError):
                self._grid[k].remove(obj)
        self._objs.remove(obj)

    def find(self, bbox):
        x0, y0, x1, y1 = bbox
        done = set()
        ret = []

        for k in self._getrange(bbox):
            if k not in self._grid:
                continue
            for obj in self._grid[k]:
                if obj in done:
                    continue
                done.add(obj)
                if obj.x1 <= x0 or x1 <= obj.x0 or obj.y1 <= y0 or y1 <= obj.y0:
                    continue
                ret.append(obj)
        return ret


def csort(objs, key):
    """Order-preserving sort with stable fallback to original index."""
    idxs = {obj: i for i, obj in enumerate(objs)}
    return sorted(objs, key=lambda obj: (key(obj), idxs[obj]))
