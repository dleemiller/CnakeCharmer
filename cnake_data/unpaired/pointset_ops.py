"""Pointset container with near-duplicate handling and geometric transforms."""

from __future__ import annotations


class pointset:
    def __init__(self):
        self.ps = []
        self.pcnt = 0

    def __str__(self):
        return "pointset of size:" + str(self.pcnt)

    def __iter__(self):
        return iter(self.ps)

    def gpscp(self, rng=None):
        if rng is None:
            rng = range(self.pcnt)
        return [self.ps[x].cp() for x in rng]

    def gps(self, rng=None):
        if rng is None:
            rng = range(self.pcnt)
        return [self.ps[x] for x in rng]

    def gpsset(self):
        uniq = []
        for p in self.ps:
            if p not in uniq:
                uniq.append(p)
        return uniq

    def ap(self, np_):
        px = self.pcnt
        self.ps.append(np_)
        self.pcnt += 1
        return px

    def aps(self, nps):
        npxs = []
        for np_ in nps:
            self.ps.append(np_)
            npxs.append(self.pcnt)
            self.pcnt += 1
        return npxs

    def fp(self, p):
        for px in range(self.pcnt):
            if self.ps[px].isnear(p):
                return px
        return -1

    def np(self, np_):
        x = self.fp(np_)
        return self.ap(np_) if x == -1 else x

    def nps(self, nps):
        npxs = []
        for np_ in nps:
            x = self.fp(np_)
            npxs.append(self.ap(np_) if x == -1 else x)
        return npxs

    def fps(self, ps):
        return [self.fp(p) for p in ps]

    def disjoint(self, o):
        for p in self.ps:
            for q in o.ps:
                if p.isnear(q):
                    return False
        return True

    def trn(self, v):
        for p in self.ps:
            p.trn(v)
        return self

    def rot(self, q):
        for p in self.ps:
            p.rot(q)
        return self

    def scl(self, o):
        for p in self.ps:
            p.scl(o)
        return self

    def uscl(self, f):
        for p in self.ps:
            p.uscl(f)
        return self
