"""Nested-sequence checks and callback wrappers."""

from __future__ import annotations


def is_nested_sequence(data):
    if not hasattr(data, "__iter__"):
        return False
    for item in data:
        if not hasattr(item, "__iter__"):
            return False
    return True


def alloc_wrapper(ctx, size, stream):
    """Assumes ctx=(alloc, free) and alloc returns a pointer/int handle."""
    alloc, _free = ctx
    try:
        return int(alloc(size, stream)), 0
    except Exception:
        return 0, 1


def free_wrapper(ctx, ptr, size, stream):
    _alloc, free = ctx
    try:
        free(ptr, size, stream)
        return 0
    except Exception:
        return 1


def get_buffer_pointer(buf, size):
    if isinstance(buf, int):
        return buf
    mv = memoryview(buf)
    if mv.nbytes != size:
        raise ValueError(f"buffer has {mv.nbytes} bytes, expected {size}")
    if mv.ndim != 1:
        raise ValueError("buffer must be 1D")
    return mv
