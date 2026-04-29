"""Quadratic-element midside node relaxation kernels."""

from __future__ import annotations


def _relax_pair(pts, mid, a, b, rfac):
    for j in range(3):
        pts[mid][j] = pts[mid][j] * (1.0 - rfac) + (pts[a][j] + pts[b][j]) * 0.5 * rfac


def relax_mid_tet(cell, pts, rfac):
    edges = [(4, 0, 1), (5, 1, 2), (6, 2, 0), (7, 0, 3), (8, 1, 3), (9, 2, 3)]
    for m, a, b in edges:
        _relax_pair(pts, cell[m], cell[a], cell[b], rfac)


def relax_mid_hex(cell, pts, rfac):
    edges = [
        (8, 0, 1),
        (9, 1, 2),
        (10, 2, 3),
        (11, 3, 0),
        (12, 4, 5),
        (13, 5, 6),
        (14, 6, 7),
        (15, 7, 4),
        (16, 0, 4),
        (17, 1, 5),
        (18, 2, 6),
        (19, 3, 7),
    ]
    for m, a, b in edges:
        _relax_pair(pts, cell[m], cell[a], cell[b], rfac)
