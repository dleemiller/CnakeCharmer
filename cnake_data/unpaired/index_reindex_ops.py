"""Index-based reindex, gather, and scatter kernels."""

from __future__ import annotations


def index_range(start, stop):
    return list(range(start, stop))


def reindex_vector(data, indices):
    if len(data) != len(indices):
        raise ValueError("shape mismatch")
    datacopy = list(data)
    for i, newi in enumerate(indices):
        if newi < 0 or newi >= len(data):
            raise IndexError("index out of bounds")
        data[i] = datacopy[newi]
    return data


def take_vector(data, indices, out=None):
    if out is None:
        out = [0.0] * len(indices)
    for i, idx in enumerate(indices):
        out[i] = data[idx]
    return out


def put_vector(out, indices, data):
    if len(data) < len(indices):
        raise ValueError("data too small")
    for i, idx in enumerate(indices):
        out[idx] = data[i]
    return out
