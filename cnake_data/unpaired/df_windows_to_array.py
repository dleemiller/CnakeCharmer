"""Convert grouped dataframe rows into padded 3D arrays."""

from __future__ import annotations

import numpy as np


def df_to_arrays_numeric(df):
    _, s, l = np.unique(df["id"].astype(int), return_index=True, return_counts=True)
    starts = s.astype(int)
    lengths = l.astype(int)
    df_values = df.iloc[:, 1:].values.astype(float)

    values = np.full((len(s), max(l), len(df.columns) - 1), np.nan)
    for i in range(len(s)):
        start = starts[i]
        end = starts[i] + lengths[i]
        values[i, 0 : lengths[i], :] = df_values[start:end, :]
    return np.array(values)
