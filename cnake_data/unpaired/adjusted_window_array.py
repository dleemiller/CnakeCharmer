"""Adjusted array with moving window traversal and in-place historical adjustments."""

from __future__ import annotations

import numpy as np


class AdjustedArray:
    def __init__(
        self, data: np.ndarray, mask: np.ndarray | None, adjustments: dict[int, list]
    ) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if mask is not None:
            m = np.asarray(mask)
            if m.shape != arr.shape:
                raise ValueError("mask shape must match data shape")
            arr = arr.copy()
            arr[~m.astype(bool)] = np.nan
        self._data = arr
        self.adjustments = adjustments

    @property
    def data(self) -> np.ndarray:
        out = np.asarray(self._data, dtype=np.float64)
        out.setflags(write=False)
        return out

    def traverse(self, window_length: int, offset: int = 0):
        if window_length < 1:
            raise ValueError("window_length must be >=1")
        if window_length > self._data.shape[0]:
            raise ValueError("window_length too long")
        return _AdjustedArrayWindow(self._data.copy(), self.adjustments, window_length, offset)


class _AdjustedArrayWindow:
    def __init__(
        self, data: np.ndarray, adjustments: dict[int, list], window_length: int, offset: int
    ) -> None:
        self.data = data
        self.window_length = window_length
        self.anchor = window_length + offset
        self.max_anchor = data.shape[0]
        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments.keys(), reverse=True)
        self.next_adj = (
            self.adjustment_indices.pop() if self.adjustment_indices else self.max_anchor
        )

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.anchor > self.max_anchor:
            raise StopIteration
        while self.next_adj < self.anchor:
            for adjustment in self.adjustments[self.next_adj]:
                adjustment.mutate(self.data)
            self.next_adj = (
                self.adjustment_indices.pop() if self.adjustment_indices else self.max_anchor
            )
        start = self.anchor - self.window_length
        out = np.asarray(self.data[start : self.anchor])
        out.setflags(write=False)
        self.anchor += 1
        return out
