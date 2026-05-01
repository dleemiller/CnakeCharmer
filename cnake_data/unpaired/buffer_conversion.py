"""Buffer protocol helpers and numpy/pyarrow conversion utilities."""

from __future__ import annotations

import numpy


def to_numpy(v):
    if isinstance(v, numpy.ndarray):
        return v
    if hasattr(v, "to_numpy"):
        return v.to_numpy()
    if hasattr(v, "as_numpy"):
        return v.as_numpy()
    return numpy.array(v, copy=False)


def to_pyarrow(v, pyarrow_mod):
    if isinstance(v, pyarrow_mod.Array):
        return v
    return pyarrow_mod.array(v)
