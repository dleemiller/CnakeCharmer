"""Compact ragged alignment array with flat storage."""

from __future__ import annotations

import numpy as np


class AlignmentArray:
    def __init__(self, alignment):
        self._starts_ends = np.zeros(len(alignment) + 1, dtype="int32")

        data_len = 0
        for idx, outer in enumerate(alignment):
            outer_len = len(outer)
            self._starts_ends[idx + 1] = self._starts_ends[idx] + outer_len
            data_len += outer_len

        self._lengths = None
        self._data = np.empty(data_len, dtype="int32")

        idx = 0
        for outer in alignment:
            for inner in outer:
                self._data[idx] = inner
                idx += 1

    def __getitem__(self, idx):
        starts = self._starts_ends[:-1]
        ends = self._starts_ends[1:]
        if isinstance(idx, int):
            start = starts[idx]
            end = ends[idx]
        elif isinstance(idx, slice):
            if not (idx.step is None or idx.step == 1):
                raise ValueError("slice step must be 1")
            start = starts[idx]
            if len(start) == 0:
                return self._data[0:0]
            start = start[0]
            end = ends[idx][-1]
        else:
            raise ValueError("invalid index type")
        return self._data[start:end]

    @property
    def data(self):
        return self._data

    @property
    def lengths(self):
        if self._lengths is None:
            self._lengths = self.ends - self.starts
        return self._lengths

    @property
    def ends(self):
        return self._starts_ends[1:]

    @property
    def starts(self):
        return self._starts_ends[:-1]
