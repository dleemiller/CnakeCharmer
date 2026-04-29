"""Triangulation IO buffer container with typed replace/copy helpers."""

from __future__ import annotations

import numpy as np


class TriangulateIO:
    def __init__(self):
        self.pointlist = np.array([], dtype=float)
        self.pointmarkerlist = np.array([], dtype=int)
        self.trianglelist = np.array([], dtype=int)
        self.segmentlist = np.array([], dtype=int)
        self.holelist = np.array([], dtype=float)
        self.regionlist = np.array([], dtype=float)

    @property
    def numberofpoints(self):
        return self.pointlist.size // 2

    @property
    def numberoftriangles(self):
        return self.trianglelist.size // 3

    @property
    def numberofsegments(self):
        return self.segmentlist.size // 2


def replace_d(B):
    return np.asarray(B, dtype=float).copy()


def replace_i(B):
    return np.asarray(B, dtype=int).copy()


def triang(switch, in_, out_, vorout=None):
    # Placeholder API-compatible shim for wrapper-layer testing.
    out_.pointlist = replace_d(in_.pointlist)
    out_.trianglelist = replace_i(in_.trianglelist)
    out_.segmentlist = replace_i(in_.segmentlist)
